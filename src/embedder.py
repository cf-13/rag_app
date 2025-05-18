import os
import json
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

class TextEmbedder:
    def __init__(self, input_dir: str, output_dir: str, embedding_model: str = "mock"):
        """
        Initialize the text embedder.
        
        Args:
            input_dir: Directory containing chunked documents
            output_dir: Directory to save embedded documents
            embedding_model: Model to use for embeddings (mock for demonstration)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        self.embedding_dim = 384  # Standard dimension for simple embeddings
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initialized TextEmbedder with {embedding_model} model")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate mock embeddings for demonstration purposes.
        
        For a real application, this would be replaced with actual embeddings 
        from a proper language model, but for demonstration we'll create deterministic
        pseudo-random embeddings based on text content.
        """
        # Create a deterministic seed from the text
        hash_obj = hashlib.md5(text.encode('utf-8'))
        seed = int(hash_obj.hexdigest(), 16) % (2**32)
        
        # Generate a deterministic embedding vector
        rng = np.random.RandomState(seed)
        embedding = rng.normal(0, 1, self.embedding_dim).astype(np.float32)
        
        # Normalize to unit length (cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding.tolist()
    
    def process_all_chunks(self) -> None:
        """Process all chunks in the input directory."""
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.json'):
                input_path = os.path.join(self.input_dir, filename)
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                    
                    # Get embedding for the chunk text
                    chunk_text = chunk['chunk_text']
                    embedding = self.get_embedding(chunk_text)
                    
                    # Add embedding to the chunk
                    chunk['embedding'] = embedding
                    
                    # Save embedded chunk
                    output_path = os.path.join(self.output_dir, filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
                    print(f"Embedded and saved: {filename}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

def main():
    input_dir = "chunked_data"
    output_dir = "embedded_data"
    embedding_model = "mock"  # Using mock embeddings for demonstration
    
    embedder = TextEmbedder(input_dir, output_dir, embedding_model)
    embedder.process_all_chunks()
    
if __name__ == "__main__":
    main()