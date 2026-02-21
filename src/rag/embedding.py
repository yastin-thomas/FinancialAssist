import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def create_vector_db(docs_dir="financial_docs", db_dir="financial_db"):
    """
    Ingest financial documents into a FAISS vector database.
    """
    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} not found. Run scrape_data.py first.")
        return

    # Check if API key is present
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    print("Loading documents...")
    # Using DirectoryLoader to recursively load .txt files
    # Note: glob="**/*.txt" is usually recursive by default in DirectoryLoader
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    
    if not docs:
        print("No documents found.")
        return
        
    print(f"Loaded {len(docs)} documents.")

    # Post-process docs to extract metadata from content
    for doc in docs:
        lines = doc.page_content.split('\n')
        for line in lines[:5]: # Check first 5 lines
            if line.startswith("Source: "):
                url = line.replace("Source: ", "").strip()
                doc.metadata['source'] = url
                break

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} text chunks.")
    
    print("Creating embeddings and vector store (FAISS)...")
    
    # FAISS does not need to remove the directory beforehand like Chroma, 
    # but we will overwrite the save by just saving again.
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Save locally
    vectorstore.save_local(db_dir)
    print(f"Vector database created successfully in {db_dir}")

if __name__ == "__main__":
    create_vector_db()
