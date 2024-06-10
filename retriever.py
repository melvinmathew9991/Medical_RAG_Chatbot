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
    """
    Class for retrieving documents based on similarity search.

    Attributes:
        loader (TextLoader): The text loader instance for loading documents.
        documents (list): The list of loaded documents.
        db (FAISS): The FAISS vector store for similarity search.
    """

    def __init__(self):
        """
        Initialize the DocumentRetriever instance.
        """
        self.loader = TextLoader("./speech.txt")  
        self.documents = self.loader.load()  

        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)  
        self.documents = splitter.split_documents(self.documents)  # Split documents into chunks

        embeddings = OpenAIEmbeddings()  
        self.db = FAISS.from_documents(self.documents, embeddings)  
    def query(self, query_text):
        """
        Perform a similarity search for the given query text.

        Args:
            query_text (str): The query text for similarity search.

        Returns:
            list: A list of documents similar to the query, or an empty list if no matches are found.
        """
        try:
            start_time = time.time()  
            processed_query = query_text.strip()  
            results = self.db.similarity_search(processed_query)  # Perform similarity search
            end_time = time.time() 
            print("Time taken for document query:", end_time - start_time, "seconds") 
            return results  
        except Exception as e:
            # Handle any exceptions that occur during document query
            print("Error occurred during document query:") 
            print(traceback.format_exc()) 
            return []  

retriever = DocumentRetriever()  