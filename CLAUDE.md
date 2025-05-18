# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Setup: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- Run app: `python src/app.py --query "Your question here"`
- Run crawler: `python src/crawler.py`
- Run chunker: `python src/chunker.py`
- Run embedder: `python src/embedder.py`
- Run vector DB: `python src/vector_db.py`

## Code Style
- Imports: Standard library first, third-party next, local modules last
- Type hints: Always use typing annotations
- Functions: Doc comments with Args/Returns
- Error handling: Use try/except with specific exceptions
- Naming: snake_case for functions/variables, CamelCase for classes
- Indentation: 4 spaces
- Line length: Maximum 100 characters
- String quotes: Single quotes for simple strings, double for docstrings
- Japanese text: Use UTF-8 encoding with comments in Japanese where necessary