from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import traceback
import os
from dotenv import load_dotenv
import time

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class DocumentRetriever:
    def __init__(self):
        self.loader = TextLoader("./speech.txt")
        self.documents = self.loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        self.documents = splitter.split_documents(self.documents)

        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(self.documents, embeddings)

    def query(self, query_text):
        try:
            start_time = time.time()
            processed_query = query_text.strip()
            results = self.db.similarity_search(processed_query)
            end_time = time.time()
            print("Time taken for document query:", end_time - start_time, "seconds")
            return results
        except Exception as e:
            print("Error occurred during document query:")
            print(traceback.format_exc())
            return []

retriever = DocumentRetriever()
