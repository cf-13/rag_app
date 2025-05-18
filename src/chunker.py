import json
import os
from typing import List, Dict, Any
import re

class DocumentChunker:
    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document chunker.
        
        Args:
            input_dir: Directory containing crawled JSON documents
            output_dir: Directory to save chunked documents
            chunk_size: Maximum token size for each chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def process_all_documents(self) -> None:
        """Process all documents in the input directory."""
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.json'):
                input_path = os.path.join(self.input_dir, filename)
                try:
                    with open(input_path, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                    
                    chunked_docs = self.chunk_document(document)
                    
                    # Save chunked documents
                    output_basename = os.path.splitext(filename)[0]
                    for i, chunk in enumerate(chunked_docs):
                        output_filename = f"{output_basename}_chunk_{i}.json"
                        output_path = os.path.join(self.output_dir, output_filename)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(chunk, f, ensure_ascii=False, indent=2)
                    
                    print(f"Processed {filename} into {len(chunked_docs)} chunks")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on chunk size and overlap.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of chunked documents with metadata
        """
        text = document['text']
        
        # Simple chunking by approximate token count (using words as proxy)
        words = re.findall(r'\w+', text)
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            # Get chunk words
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                continue
                
            # Reconstruct text (this is an approximation)
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk with metadata
            chunk = {
                'chunk_text': chunk_text,
                'metadata': {
                    'source_url': document['url'],
                    'title': document['title'],
                    'categories': document.get('categories', []),
                    'chunk_id': len(chunks),
                    'original_metadata': document.get('metadata', {})
                }
            }
            
            chunks.append(chunk)
        
        return chunks

def main():
    input_dir = "crawled_data"
    output_dir = "chunked_data"
    
    chunker = DocumentChunker(input_dir, output_dir)
    chunker.process_all_documents()
    
if __name__ == "__main__":
    main()