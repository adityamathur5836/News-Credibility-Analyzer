"""
RAG Pipeline: Embedding and retrieving evidence using FAISS.
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

class RAGPipeline:
    """Manages in-memory FAISS indexing and retrieval."""
    def __init__(self):
        # We use a fast, small, local HuggingFace embedding model
        # Requires sentence-transformers
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        
    def build_index(self, texts, metadatas=None):
        """Build an in-memory FAISS vector store from given texts."""
        if not texts:
            return
            
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            docs.append(Document(page_content=text, metadata=meta))
            
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        
    def retrieve_evidence(self, query, top_k=3):
        """Retrieve most relevant chunks for a query."""
        if not self.vector_store:
            return []
            
        # retrieve relevant docs
        docs = self.vector_store.similarity_search(query, k=top_k)
        return docs
