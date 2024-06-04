from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

loader = TextLoader("./speech.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
documents = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def query(query_text):
    try:
        processed_query = query_text.strip()
        return db.similarity_search(processed_query)
    except Exception as e:
        print("Error occurred during document query:")
        print(traceback.format_exc())
        return []
