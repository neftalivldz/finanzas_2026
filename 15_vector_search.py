# Databricks notebook source
Basado en:
https://www.kaggle.com/datasets/arpansri/books-summary$0
https://awadrahman.medium.com/showcasing-databricks-vector-search-a-hands-on-example-1-d393c40dd3d4$0
https://docs.databricks.com/aws/en/vector-search/vector-search-python-sdk-example$0


# COMMAND ----------

df = spark.read.format("csv") \
    .option("header", "true") \
    .option("quote", '"') \
    .option("escape", '"') \
    .load("/Volumes/itesm/finanzas/nvda/books_summary.csv")

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table itesm.finanzas.dataset

# COMMAND ----------

from delta import *
df \
.withColumnRenamed("_c0", "index")\
.withColumnRenamed("book_name", "title")\
.withColumnRenamed("summaries", "summary")\
.withColumnRenamed("categories", "genre")\
.write.format("delta").mode("overwrite").saveAsTable("itesm.finanzas.dataset")

# COMMAND ----------

raw_table_name="itesm.finanzas.dataset"
raw_table = spark.read.table(raw_table_name)
raw_table.printSchema()

# COMMAND ----------

table_clean= raw_table.filter("index is NOT NULL AND\
                                index RLIKE '^[0-9]+$' AND\
                                title IS NOT NULL AND\
                                size(split(title, ' ')) <= 30 AND\
                                summary IS NOT NULL AND")
errors=raw_table.subtract(table_clean)
print("Good Quality Rows:",table_clean.count())
print("Low Quality Rows:" ,errors.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table itesm.finanzas.books_errors

# COMMAND ----------

errors_table_name="itesm.finanzas.books_errors"
errors.write.format("delta").mode("overwrite").saveAsTable(errors_table_name)

# COMMAND ----------

table_clean = table_clean.dropDuplicates(["index", "title"])
print("Clean Table:",table_clean.count(),"out of:", raw_table.count())

# COMMAND ----------

from pyspark.sql import functions as F

combined = table_clean.withColumn(
    "combined_text",
    F.concat(
        F.lit("Title: "), "title", F.lit("\n"),
        F.lit("Genre: "), "genre", F.lit("\n"),
        F.lit("Summary: "), "summary", F.lit("\n")
    )
)
combined.printSchema()

# COMMAND ----------

combined.write.saveAsTable("itesm.finanzas.combined",mode="overwrite")


# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

df_with_length = combined.withColumn('text_length', F.length(combined['combined_text']))\
                                .withColumn('token_length', (F.length(combined['combined_text'])/75*100).cast(IntegerType()))\
                                .select("index", "title","text_length","token_length")

max_input_tokens = 8191  # open AI max tocken check:  https://platform.openai.com/docs/guides/embeddings/embedding-models
long_text= df_with_length.filter(df_with_length['token_length'] > max_input_tokens)
print("Number of texts exceeding max token:",long_text.count())

# COMMAND ----------

display(df_with_length.limit(5))

# COMMAND ----------

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

# COMMAND ----------

print(tokenizer.encode("Vector Search is smart!"))
print(tokenizer.encode("Vector Search is brilliant!"))

# COMMAND ----------

tokenizer.decode([3866, 7694, 374, 7941, 0])


# COMMAND ----------

combined_table_name = "itesm.finanzas.combined"
print("Reading Table:",combined_table_name)

combined_table= spark.read.table(combined_table_name)
combined_table.printSchema()
print("num of rows in ",combined_table_name ,combined_table.count())

# COMMAND ----------

def chunk_text(text, max_chunk_tokens = 1024):
    tokens = tokenizer.encode(text)
    chunked_text = []
    while tokens:
        chunk_tokens = tokens[:max_chunk_tokens]
        chunk_text  = tokenizer.decode(chunk_tokens)
        chunked_text.append(chunk_text)
        tokens = tokens[max_chunk_tokens:]
    return chunked_text

# COMMAND ----------

combined_pandas

# COMMAND ----------

combined_pandas = combined_table.toPandas()
combined_pandas = combined_pandas.dropna(subset=['combined_text'])
processed_data = []
for _ , row in combined_pandas.iterrows():
    text_chunks = chunk_text(row['combined_text'])
    chunk_no = 0
    for chunk in text_chunks:
        row_data = row.to_dict()
        row_data['index'] = f"{row['index']}:{chunk_no}"
        row_data['combined_text'] = chunk
        processed_data.append(row_data)
        chunk_no += 1

# COMMAND ----------

import pandas as pd
chunked_pandas_df = pd.DataFrame(processed_data)
chunked_spark_df = spark.createDataFrame(chunked_pandas_df)
print("combined table length  :", combined_table.count())
print("chunked table length  :", chunked_spark_df.count())
chunked_spark_df.display()

# COMMAND ----------

chunked_table_name = "itesm.finanzas.chunked_data" 
chunked_spark_df.write.format("delta").mode("overwrite").saveAsTable(chunked_table_name)

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# COMMAND ----------

help(VectorSearchClient)


# COMMAND ----------

# MAGIC %md
# MAGIC # Create Vector Search Endpoint
# MAGIC

# COMMAND ----------

vector_search_endpoint_name = "books-vs-edpoint"


# COMMAND ----------

vsc.create_endpoint(
    name=vector_search_endpoint_name,
    endpoint_type="STANDARD" # or "STORAGE_OPTIMIZED"
)

# COMMAND ----------

endpoint = vsc.get_endpoint(name=vector_search_endpoint_name)
endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC #Create index

# COMMAND ----------

embedding_model_endpoint = "databricks-qwen3-embedding-0-6b"
vs_index_fullname        = "itesm.finanzas.books_openai_index"
source_table_fullname        = "itesm.finanzas.chunked_data"

print("endpoint_name                :", endpoint)
print("source_table_fullname        :", source_table_fullname)
print("index_name                   :", vs_index_fullname)
print("embedding_model_endpoint_name:", embedding_model_endpoint)

# COMMAND ----------

spark.sql("ALTER TABLE itesm.finanzas.chunked_data SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# DBTITLE 1,Cell 32
import time
from IPython.display import clear_output

start_time = time.time()  

index = vsc.create_delta_sync_index(
  endpoint_name=vector_search_endpoint_name,
  source_table_name=source_table_fullname,
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="index",
  embedding_source_column="combined_text",
  embedding_model_endpoint_name=embedding_model_endpoint
)
index.describe()

# COMMAND ----------

index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=vs_index_fullname)

index.describe()

# COMMAND ----------

# Wait for index to come online. Expect this command to take several minutes.
import time
while not index.describe().get('status').get('detailed_state').startswith('ONLINE'):
  print("Waiting for index to be ONLINE...")
  time.sleep(5)
print("Index is ONLINE")
index.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #Using it up

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vs_client = VectorSearchClient()

# COMMAND ----------

vs_endpoint_name="books-vs-edpoint"
index_name = "itesm.finanzas.books_openai_index"

index = vs_client.get_index(endpoint_name= vs_endpoint_name,
                            index_name= index_name)
index.describe()['status']['message'] 

# COMMAND ----------

query="the dark in california civil war"
results = index.similarity_search(
  query_text=query,
  columns=["title","summary"],
  num_results=5
  )

# COMMAND ----------

print(type(results))
print(results.keys())
print(results["result"].keys())
print(len(results["result"]["data_array"]))
print(len(results["result"]["data_array"][0]))

# COMMAND ----------

rows = results['result']['data_array']
for i, (title, summary, score) in enumerate(rows):
  if len(summary) > 50:
    # trim text output for readability
    summary = summary[0:50] + "..."
  print(f" Title: {title}  Score: {score}")

# COMMAND ----------

review_book="An historic fiction about the fated trek of the English team to \
              the South Pole, the story is spellbinding and enables the reader \
               to experience the challenges of this journey through the eyes,  \
               ears, thoughts and memories of different members of the team. \
               While I personally would never be intrigued enough to make \
                such a dangerous and difficult trip, the writing gave me insight into the kinds of \
               people who are so motivated and instilled in me great respect  \
              for their courage, stamina and willingness to help each other. \
              Highly recommend it."

# COMMAND ----------

# Book: The Birthday Boys    
results = index.similarity_search(
  query_text=review_book,
  columns=["title","summary"],
  num_results=10
  )
rows = results['result']['data_array']
for (title, summary, score) in rows:
  if len(summary) > 100:
    # trim text output for readability
    summary = summary[0:100] + "..."
  print(f"Title: {title:30} with Score: {score}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Recommender system

# COMMAND ----------

raw_table_pd = raw_table.toPandas()
random_row= raw_table_pd.iloc[24]
bought_book_title=random_row["title"]
bought_book_summary=random_row["summary"]
print("Title:",bought_book_title)
print("Summary:",bought_book_summary[:500],"...")

# COMMAND ----------

results = index.similarity_search(
  query_text=bought_book_summary,
  columns=["title","summary"],
  num_results=5
  )

rows = results['result']['data_array']
print('The similar books to: "', bought_book_title, '":')
print("---")

for (title, summary, score) in rows:
  if len(summary) > 100:
    # trim text output for readability
    summary = summary[0:100] + "..."
  print(f"Title: {title:30} with Score: {score}")

# COMMAND ----------

from langchain_core.documents import Document
from typing import List

def convert_vector_search_to_documents(results) -> List[Document]:
  column_names = []
  for column in results["manifest"]["columns"]:
      column_names.append(column)

  langchain_docs = []
  for item in results["result"]["data_array"]:
      metadata = {}
      score = item[-1]
      # print(score)
      i = 1
      for field in item[1:-1]:
          # print(field + "--")
          metadata[column_names[i]["name"]] = field
          i = i + 1
      doc = Document(page_content=item[0], metadata=metadata)  # , 9)
      langchain_docs.append(doc)
  return langchain_docs

langchain_docs = convert_vector_search_to_documents(results)

langchain_docs

# COMMAND ----------

