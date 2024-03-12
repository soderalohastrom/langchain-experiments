import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.vectorstores.pgvector import PGVector
from pgvector_service import PgvectorService
import os
import time

load_dotenv()


# --------------------------------------------------------------
# Load the documents
# --------------------------------------------------------------

loader = TextLoader(
    "../data/The Project Gutenberg eBook of A Christmas Carol in Prose; Being a Ghost Story of Christmas.txt"
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

query = "The Project Gutenberg eBook of A Christmas Carol in Prose; Being a Ghost Story of Christmas"


"""
First, we compare to Pinecone, a managed vector store service.

"""


# 
# --------------------------------------------------------------
# Create a PGVector Store
# --------------------------------------------------------------

"""
Donwload postgresql to run locally:
https://www.postgresql.org/download/

How to install the pgvector extension:
https://github.com/pgvector/pgvector

Fix common installation issues:
https://github.com/pgvector/pgvector?tab=readme-ov-file#installation-notes
"""

COLLECTION_NAME = "The Project Gutenberg eBook of A Christmas Carol in Prose"

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "pgvector"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postres"),
)

# create the store
db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,
)

# load the store
pgvector_docsearch = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)

# --------------------------------------------------------------
# Query the index with PGVector
# --------------------------------------------------------------


def run_query_pgvector(docsearch, query):
    docs = docsearch.similarity_search(query, k=4)
    result = docs[0].page_content
    return result


calculate_average_execution_time(
    run_query_pgvector, docsearch=pgvector_docsearch, query=query
)


# --------------------------------------------------------------
# Add more collections to the database
# --------------------------------------------------------------

loader = TextLoader("../data/The Project Gutenberg eBook of Romeo and Juliet.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
new_docs = text_splitter.split_documents(documents)


COLLECTION_NAME_2 = "The Project Gutenberg eBook of Romeo and Juliet"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=new_docs,
    collection_name=COLLECTION_NAME_2,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False,
)


# --------------------------------------------------------------
# Query the index with multiple collections
# --------------------------------------------------------------

pg = PgvectorService(CONNECTION_STRING)


def run_query_multi_pgvector(docsearch, query):
    docs = docsearch.custom_similarity_search_with_scores(query, k=4)
    result = docs[0][0].page_content
    print(result)


run_query_multi_pgvector(pg, query)

# --------------------------------------------------------------
# Delete the collection
# --------------------------------------------------------------
pg.delete_collection(COLLECTION_NAME)
pg.delete_collection(COLLECTION_NAME_2)

# --------------------------------------------------------------
# Update the collection
# --------------------------------------------------------------
pg.update_collection(docs=docs, collection_name=COLLECTION_NAME)
