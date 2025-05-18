import os
import json
from typing import List, Dict, Any
import argparse

class RAGApplication:
    def __init__(self, embedding_model: str = "mock"):
        """
        Initialize the RAG application.
        
        Args:
            embedding_model: Model to use for embeddings (openai, sbert, mock, etc.)
        """
        self.embedding_model = embedding_model
        
        # Initialize components
        self.embedder = self._initialize_embedder()
        self.vector_db = self._initialize_vector_db()
        self.llm = self._initialize_llm()
    
    def _initialize_embedder(self):
        """Initialize the text embedder client."""
        if self.embedding_model == "openai":
            try:
                from openai import OpenAI
                client = OpenAI()
                print("Initialized OpenAI embedding client")
                return client
            except ImportError:
                print("Error: OpenAI Python package not installed. Run: pip install openai")
                raise
        elif self.embedding_model == "sbert":
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("Initialized SBERT embedding client")
                return model
            except ImportError:
                print("Error: Sentence-Transformers package not installed. Run: pip install sentence-transformers")
                raise
        elif self.embedding_model == "mock":
            # No client needed for mock embeddings
            print("Using mock embeddings (for demonstration)")
            return None
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def _initialize_vector_db(self):
        """Initialize the vector database client."""
        # Import our simple vector database
        from vector_db import SimpleVectorDatabase
        
        # Create and load the database
        db = SimpleVectorDatabase("embedded_data")
        db.load_documents()
        print("Initialized SimpleVectorDatabase")
        return db
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        # For demonstration, we'll use a simple mock LLM
        def mock_llm(prompt):
            """
            Simulate an LLM response for demonstration purposes.
            In a real system, this would be replaced with an actual LLM call.
            """
            # Extract the user query from the prompt
            import re
            query_match = re.search(r'ユーザーの質問: (.*?)\n', prompt)
            query = query_match.group(1) if query_match else "不明な質問"
            
            # Find referenced documents
            from collections import Counter
            words = Counter(re.findall(r'[\w\.-]+', prompt.lower()))
            
            # Generate a simple response based on query and available context
            if 'リスト' in query:
                return f"""
Pythonのリスト操作について説明します。

リストはPythonの基本的なデータ構造で、順序付けられた要素の集合です。リストは変更可能（mutable）で、異なる型の要素を含むことができます。

主なリスト操作:
1. 作成: `my_list = [1, 2, 3]`
2. 追加: `my_list.append(4)`
3. 挿入: `my_list.insert(0, 'start')`
4. 削除: `my_list.remove(2)` または `del my_list[0]`
5. インデックス参照: `my_list[0]`
6. スライス: `my_list[1:3]`
7. 結合: `my_list + [5, 6]`
8. リスト内包表記: `[x*2 for x in my_list]`

詳細な情報は Python公式ドキュメントをご参照ください。
"""
            elif 'クラス' in query:
                return "Pythonのクラスについての説明です..."
            elif 'モジュール' in query:
                return "Pythonのモジュールシステムについての説明です..."
            else:
                return f"「{query}」についての回答です。Python公式ドキュメントを参照して詳細を確認してください。"
        
        print("Using mock LLM for demonstration")
        return mock_llm
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using the selected model."""
        if self.embedding_model == "openai":
            response = self.embedder.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        elif self.embedding_model == "sbert":
            return self.embedder.encode(text).tolist()
        elif self.embedding_model == "mock":
            # Create a deterministic mock embedding using the same approach as in embedder.py
            import hashlib
            import numpy as np
            
            # Create a deterministic seed from the text
            hash_obj = hashlib.md5(text.encode('utf-8'))
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            
            # Generate a deterministic embedding vector
            rng = np.random.RandomState(seed)
            embedding = rng.normal(0, 1, 384).astype(np.float32)  # 384 dimensions
            
            # Normalize to unit length (cosine similarity)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding.tolist()
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")
    
    def search_similar_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        # Get embedding for the query
        query_embedding = self.get_embedding(query)
        
        # Search in vector database
        return self.vector_db.search(query_embedding, limit)
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the LLM with retrieved documents as context.
        
        Args:
            query: The user query
            context_docs: List of retrieved documents to use as context
            
        Returns:
            Generated response from the LLM
        """
        # Format context from retrieved documents
        context = "\n\n".join([
            f"Document: {doc['metadata'].get('title', 'Untitled')}\n"
            f"URL: {doc['metadata'].get('source_url', 'No URL')}\n"
            f"Content: {doc['text']}"
            for doc in context_docs
        ])
        
        # Create prompt for the LLM
        prompt = f"""あなたはヘルプページアシスタントです。以下の情報を参考にして、ユーザーの質問に丁寧に答えてください。

ユーザーの質問: {query}

参考情報:
{context}

上記の情報に基づき、簡潔かつ正確に回答してください。不明な点がある場合は、正直にわからないと伝えてください。参考情報の出典（URL）を回答の最後に明記してください。
        """
        
        # Generate response using the LLM
        if hasattr(self.llm, 'chat'):
            # OpenAI client
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたはヘルプページアシスタントです。正確で役立つ情報を提供してください。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        else:
            # Local LLM function
            return self.llm(prompt)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with the response and retrieved documents
        """
        # 1. Retrieve similar documents
        similar_docs = self.search_similar_documents(query)
        
        # 2. Generate response using LLM
        response = self.generate_response(query, similar_docs)
        
        # 3. Return result
        return {
            "query": query,
            "response": response,
            "retrieved_documents": similar_docs
        }

def main():
    parser = argparse.ArgumentParser(description="RAG Application for Help Pages")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--embedding-model", type=str, default="mock", 
                        choices=["openai", "sbert", "mock"], help="Embedding model to use")
    
    args = parser.parse_args()
    
    app = RAGApplication(embedding_model=args.embedding_model)
    
    if args.query:
        result = app.process_query(args.query)
        print("\n" + "="*50)
        print("Query:", result["query"])
        print("-"*50)
        print("Response:", result["response"])
        print("-"*50)
        print("Retrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\n[{i+1}] {doc['metadata'].get('title', 'Untitled')}")
            print(f"URL: {doc['metadata'].get('source_url', 'No URL')}")
            print(f"Relevance Score: {1 - doc.get('distance', 0):.4f}")
    else:
        print("No query provided. Use --query to ask a question.")

if __name__ == "__main__":
    main()