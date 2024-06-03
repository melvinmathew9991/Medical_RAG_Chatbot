from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Load medical data (modify path and loader if needed)
loader = TextLoader("./speech.txt")
documents = loader.load()

# Split documents into sentences
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
documents = splitter.split_documents(documents)

# Create embeddings with OpenAI 
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def query(query_text):
  """
  Searches the document database for similar documents based on the query.

  Args:
      query_text: The text to search for.

  Returns:
      A list of document IDs and their similarity scores to the query.
  """
  processed_query = query_text.strip()
  return db.similarity_search(processed_query)



