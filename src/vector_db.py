import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from glob import glob

class SimpleVectorDatabase:
    def __init__(self, data_dir: str):
        """
        Initialize a simple in-memory vector database.
        
        Args:
            data_dir: Directory containing embedded documents
        """
        self.data_dir = data_dir
        self.documents = []
        print("Initialized simple in-memory vector database")
    
    def load_documents(self) -> None:
        """Load all embedded documents into the in-memory database."""
        # Get all embedded document files
        files = glob(os.path.join(self.data_dir, "*.json"))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                if 'embedding' not in document:
                    print(f"Warning: No embedding found in {file_path}")
                    continue
                
                # Extract necessary data
                document_id = os.path.basename(file_path)
                embedding = document['embedding']
                text = document['chunk_text']
                metadata = document['metadata']
                
                # Add to in-memory database
                self.documents.append({
                    'id': document_id,
                    'embedding': embedding,
                    'text': text,
                    'metadata': metadata
                })
                
                print(f"Added document to vector database: {document_id}")
                
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")
                
        print(f"Loaded {len(self.documents)} documents into the vector database")
    
    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query embedding.
        
        Args:
            query_embedding: The embedding vector of the query
            limit: Maximum number of results to return
            
        Returns:
            List of similar documents with their metadata
        """
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        
        # Calculate cosine similarity between query and all documents
        results = []
        for doc in self.documents:
            doc_vector = np.array(doc['embedding'])
            # Cosine similarity calculation
            similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
            # Cosine distance = 1 - similarity (lower is better)
            distance = 1 - similarity
            
            results.append({
                'id': doc['id'],
                'text': doc['text'],
                'metadata': doc['metadata'],
                'distance': distance
            })
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x['distance'])
        
        # Return top results
        return results[:limit]

def main():
    data_dir = "embedded_data"
    
    # Use simple in-memory vector database
    db = SimpleVectorDatabase(data_dir)
    db.load_documents()
    
    # Simple test query with a mock embedding
    print("Testing vector database with a mock query...")
    import numpy as np
    
    # Generate a random embedding vector for testing
    query_embedding = np.random.normal(0, 1, 384).astype(np.float32)
    # Normalize to unit length
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Search for similar documents
    results = db.search(query_embedding.tolist())
    
    if results:
        print(f"Found {len(results)} similar documents")
        
        # Print the top result
        top_result = results[0]
        print(f"\nTop result:")
        print(f"Title: {top_result['metadata'].get('title', 'Untitled')}")
        print(f"URL: {top_result['metadata'].get('source_url', 'No URL')}")
        print(f"Similarity score: {1 - top_result.get('distance', 0):.4f}")
        
        # Print a snippet of the content
        content = top_result['text']
        snippet = content[:150] + "..." if len(content) > 150 else content
        print(f"Content snippet: {snippet}")
    
if __name__ == "__main__":
    main()