## Hybrid Search using Pinecone + LangChain

A Streamlit-based Hybrid Search Application that combines Dense (semantic) and Sparse (keyword-based) retrieval using Pinecone and LangChain for highly accurate search results.

#### Features
- Hybrid Search (Dense + Sparse Retrieval)
- Semantic search using HuggingFace embeddings
- Keyword search using BM25 (sparse encoder)
- Fast vector search with Pinecone
- Add custom documents dynamically
- Interactive UI with Streamlit
- Tech Stack
  - Frontend/UI: Streamlit
  - Vector DB: Pinecone (Serverless)
  - Framework: LangChain
  - Embeddings: HuggingFace (all-MiniLM-L6-v2)
  - Sparse Retrieval: BM25 Encoder
  - Language: Python

#### What is Hybrid Search?

Hybrid search combines:

- Dense Search (Semantic): Understands meaning of the query
- Sparse Search (Keyword): Matches exact words

This improves both accuracy + relevance compared to using either method alone.

#### Project Architecture
- User Query
   ↓
- Hybrid Retriever
(Dense + Sparse)
   ↓
- Pinecone Index
   ↓
- Top Relevant Documents
   ↓
- Displayed in UI

#### How It Works

- Add Documents :-
Enter multiple lines of text
Each line is treated as a separate document

- Indexing :-
Dense embeddings generated using HuggingFace
BM25 encoder trained for sparse retrieval
Data stored in Pinecone index

- Querying :-
User enters a search query
Hybrid retriever combines:
Semantic similarity
Keyword matching
Returns most relevant results

#### Key Components
- Pinecone Index: Stores vector embeddings
- BM25 Encoder: Handles keyword-based scoring
- Hybrid Retriever: Combines dense + sparse search
- Streamlit UI: Provides interaction layer
