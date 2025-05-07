# Databricks notebook source
# DBTITLE 1,Install required external libraries
# MAGIC %pip install --quiet -U mlflow[databricks] lxml==4.9.3 transformers==4.49.0 langchain==0.3.19 databricks-vectorsearch==0.49 bs4==0.0.2 markdownify==0.14.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

!python3 --version

# COMMAND ----------

# DBTITLE 1,Init our resources and catalog
# MAGIC %run ./00-Init $reset_all_data=false

# COMMAND ----------

import markdownify
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import requests
# Fetch the XML content from sitemap
response = requests.get(DATABRICKS_SITEMAP_URL)
root = ET.fromstring(response.content)
max_documents=None
# Find all 'loc' elements (URLs) in the XML
urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
if max_documents:
  urls = urls[:max_documents]

    # Create DataFrame from URLs
df_urls = spark.createDataFrame(urls, StringType()).toDF("url").repartition(10)
df_urls = df_urls.filter(df_urls.url.startswith("https://www.ford.com/finance/customer-support"))
#df_urls = df_urls.limit(1)

    # Pandas UDF to fetch HTML content for a batch of URLs
@pandas_udf("string")
def fetch_html_udf(urls: pd.Series) -> pd.Series:
  adapter = HTTPAdapter(max_retries=retries)
  http = requests.Session()
  http.mount("http://", adapter)
  http.mount("https://", adapter)
  def fetch_html(url):
    try:
      response = http.get(url)
      if response.status_code == 200:
        return response.content
    except requests.RequestException:
      return None
    return None

  with ThreadPoolExecutor(max_workers=200) as executor:
    results = list(executor.map(fetch_html, urls))
    return pd.Series(results)

    # Pandas UDF to process HTML content and extract text
@pandas_udf("string")
def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
  def extract_text(html_content):
    if html_content:
      soup = BeautifulSoup(html_content, "html.parser")
      article = soup.find("div", {"class": "astute-onetopic-result-n"})
      if article:
        try:
          return markdownify.markdownify(article.prettify(), heading_style="ATX")
        except Exception as e:
          return None
    return None

  return html_contents.apply(extract_text)



# COMMAND ----------

    # Apply UDFs to DataFrame
df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
display(df_with_html)


# COMMAND ----------

final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))
display(final_df)

# COMMAND ----------

display(spark.table("databricks_documentation_vs_index_llmops").limit(2))

# COMMAND ----------

if not spark.catalog.tableExists("ford_documentation") or spark.table("ford_documentation").isEmpty():
    # Download Databricks documentation to a DataFrame (see _resources/00-init for more details)
    doc_articles = download_databricks_documentation_articles()
    #Save them as a raw_documentation table
    doc_articles.write.mode('overwrite').saveAsTable("ford_documentation")

display(spark.table("ford_documentation").limit(2))

# COMMAND ----------

# DBTITLE 1,Splitting our html pages in smaller chunks
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 500

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "header2")])

# Split on H2, but merge small h2 chunks together to avoid having too small chunks. 
def split_html_on_h2(html, min_chunk_size=20, max_chunk_size=500):
    if not html:
        return []
    #removes b64 images captured in the md    
    html = re.sub(r'data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\n]+', '', html, flags=re.MULTILINE)
    chunks = []
    previous_chunk = ""
    for c in md_splitter.split_text(html):
        content = c.metadata.get('header2', "") + "\n" + c.page_content
        if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size / 2:
            previous_chunk += content + "\n"
        else:
            chunks.extend(text_splitter.split_text(previous_chunk.strip()))
            previous_chunk = content + "\n"
    if previous_chunk:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
    return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]

# Let's try our chunking function
html = spark.table("ford_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_documentation table containing chunks
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embeddings STRING
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# Let's create a user-defined function (UDF) to chunk all our documents with spark
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    
(spark.table("ford_documentation")
      .filter('text is not null')
      .repartition(30)
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable("databricks_documentation"))

display(spark.table("databricks_documentation"))

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS support_doc_embeddings_v2 (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embeddings ARRAY<STRING>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

import mlflow.deployments
import array
deploy_client = mlflow.deployments.get_deploy_client("databricks")

def get_embeddings(input: pd.Series):
    response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [input]})
    embeddings = [e['embedding'] for e in response.data]
    return input
# Register the Python function as a UDF
getEmbeddings = F.udf(
    lambda content: content,#get_embeddings(content),
    "array<double>",
)
docs=spark.table("databricks_documentation").withColumn("embeddings", getEmbeddings("content"))


# COMMAND ----------

import array



def get_embeddings(input):
    response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": [str(input)]})
    embeddings = [str(e['embedding']) for e in response.data]
    return embeddings

@pandas_udf("array<string>")
def test(input: pd.Series) -> pd.Series:
    result=input.apply(get_embeddings)
    return result

docs=spark.table("databricks_documentation").withColumn("embeddings", test("content"))
docs.collect()

# COMMAND ----------

docs.write.mode('overwrite').saveAsTable("support_doc_embeddings_v2")

# COMMAND ----------

# DBTITLE 1,What is an embedding
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

#Embeddings endpoints convert text into a vector (array of float). Here is an example using GTEgte:
response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)
#

# COMMAND ----------

# DBTITLE 1,Creating the Vector Search endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()
if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# DBTITLE 1,Create the managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.databricks_documentation"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.databricks_documentation_vs_index_llmops"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  try:
    vsc.create_delta_sync_index(
      endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
      index_name=vs_index_fullname,
      source_table_name=source_table_fullname,
      pipeline_type="TRIGGERED",
      primary_key="id",
      embedding_source_column='content', #The column containing our text
      embedding_model_endpoint_name='databricks-gte-large-en' #The embedding endpoint used to create the embeddings
    )
  except Exception as e:
    display_quota_error(e, VECTOR_SEARCH_ENDPOINT_NAME)
    raise e
  #Let's wait for the index to be ready and all our embeddings to be created and indexed
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Live Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also support a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "How can I track billing usage on my workspaces?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG using DBRX
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your real-time chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [02-Deploy-RAG-Chatbot-Model]($./02-Deploy-RAG-Chatbot-Model) notebook to create and deploy a chatbot endpoint.