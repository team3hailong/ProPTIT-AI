import os
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

data_dir = os.path.join(os.path.dirname(__file__), 'data')
store_dir = os.path.join(os.path.dirname(__file__), 'embeddings')
data_path = os.path.join(data_dir, 'nganh_hoc.csv')
store_path = os.path.join(store_dir, 'vectorstore.faiss')

os.makedirs(store_dir, exist_ok=True)

def build_vectorstore():
    loader = CSVLoader(file_path=data_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(store_path)
    return db

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Nếu vectorstore chưa tồn tại hoặc file CSV mới hơn vectorstore thì rebuild
    need_rebuild = False
    if not os.path.exists(store_path):
        need_rebuild = True
    else:
        csv_mtime = os.path.getmtime(data_path)
        store_mtime = os.path.getmtime(store_path)
        if csv_mtime > store_mtime:
            need_rebuild = True
    if need_rebuild:
        return build_vectorstore()
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
