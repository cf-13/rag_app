# RAG Application for Technical Support

A Retrieval-Augmented Generation (RAG) application that extracts knowledge from help pages and uses vector search to efficiently answer similar questions.

## Architecture

```
RAG Application
    ├───┬─── [User Query Embedding]
    │   ↓
    ├─── [Vector Database
    │   ↑  Similarity Search
    └─── [Help Page Data (Chunks, Embeddings, Metadata)]
            ↑ (Periodic Ingestion)
            [Crawler/Parser]
            ↑
            [Various Help Pages/Knowledge Bases]
```

## Key Components

1. **Crawler**: Collects data from help pages
2. **Chunker**: Splits long text into appropriate sizes
3. **Embedder**: Vectorizes text
4. **Vector DB**: Stores embeddings for fast search
5. **Application**: User interface and LLM integration

## Setup

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Create directories
mkdir -p crawled_data chunked_data embedded_data
```

## Usage

### 1. Crawl help pages

```bash
python src/crawler.py
```

The default configuration crawls the Python documentation. You can modify `crawler.py` to target your desired help pages.

### 2. Split text into chunks

```bash
python src/chunker.py
```

### 3. Generate embedding vectors

```bash
python src/embedder.py
```

The current implementation uses mock embeddings for demonstration purposes. For production use, replace with OpenAI or SBERT.

### 4. Load data into vector database

```bash
python src/vector_db.py
```

The current implementation uses a simple in-memory vector database. For production use, consider using ChromaDB or other vector databases.

### 5. Run the application

```bash
python src/app.py --query "How do I use Python lists?"
```

## Customization

- **Embedding Model**: Use the `--embedding-model` option to specify `openai`, `sbert`, or `mock` (default)
- **Vector DB**: The application uses a simple in-memory vector database, but can be extended to support ChromaDB or Qdrant

## Data Flow

1. Extract text data from help pages
2. Split text into appropriately sized chunks
3. Convert each chunk to embedding vectors
4. Store vectors and metadata in the database
5. Convert user queries to embeddings and perform similarity search
6. Use search results as context for the LLM
7. LLM generates appropriate answers

## Notes

This project is for demonstration purposes. For production use, consider:
- Implementing proper error handling
- Adding authentication and rate limiting
- Using a persistent vector database
- Implementing proper monitoring and logging
- Adding a web interface or API