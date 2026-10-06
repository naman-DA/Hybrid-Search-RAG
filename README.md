# Hybrid Search RAG

A production-oriented hybrid search application that combines dense semantic retrieval with sparse BM25 keyword retrieval to improve document search relevance.

The application uses HuggingFace embeddings for semantic search, BM25 for keyword-based retrieval, Pinecone as the vector database, and Streamlit for the user interface.

## 🚀 Live Demo

https://hybrid-search-rag-wxfwxnbbmzpezg6ylwas7m.streamlit.app/

## 📌 Overview

Traditional keyword search can miss documents when the query uses different wording from the source text, while purely semantic search can sometimes miss exact terms, identifiers, or keyword-specific information.

This project combines both approaches:

- Dense retrieval captures semantic similarity.
- Sparse BM25 retrieval captures keyword and lexical relevance.
- Pinecone performs hybrid retrieval using both signals.
- Hybrid scores are displayed directly in the Streamlit interface.
- Document metadata such as document ID and source is returned with each result.

The goal is to provide more robust retrieval by combining the strengths of semantic and keyword-based search.

## 🧠 How Hybrid Search Works

The application uses two complementary retrieval signals.

### 1. Dense Semantic Search

Documents are converted into vector embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

The query is also converted into an embedding. Semantic similarity between the query and documents allows the system to retrieve relevant content even when the exact query words do not appear in the document.

### 2. Sparse BM25 Search

BM25 provides lexical retrieval based on the words present in the query and documents.

This is useful for:

- Exact terminology
- Names
- Identifiers
- Keyword-heavy queries
- Queries where lexical matching is important

### 3. Hybrid Retrieval

The dense and sparse representations are combined using a configurable weighting factor.

The current implementation uses:

`alpha = 0.5`

This gives equal weight to the dense and sparse retrieval signals.

The resulting hybrid representation is queried against Pinecone, and the returned relevance score is displayed in the application.

## 🏗️ Architecture

User Query

→ Query Embedding using HuggingFace

→ BM25 Query Encoding

→ Dense + Sparse Hybrid Representation

→ Pinecone Hybrid Search

→ Top-K Relevant Documents

→ Hybrid Scores + Metadata

→ Streamlit UI

## 🔄 Document Ingestion Pipeline

Documents are processed and stored as a searchable corpus.

Document Text

→ Document Collection

→ HuggingFace Embeddings

→ Dense Vectors

+

Document Collection

→ BM25 Encoder

→ Sparse Representation

→ Pinecone

The project also persists the local document corpus in `documents.json` and the BM25 encoder values in `bm25_values.json`.

## ✨ Features

- Dense semantic search
- Sparse BM25 keyword search
- Hybrid dense + sparse retrieval
- Pinecone vector database
- HuggingFace sentence-transformer embeddings
- Persistent local document corpus
- Persistent BM25 values
- Document metadata
- Hybrid relevance scores
- Top-K result retrieval
- Streamlit web interface
- Local `.env` support
- Streamlit Cloud secrets support
- Cloud deployment

## 🛠️ Tech Stack

### Programming

- Python

### Generative AI / Retrieval

- Hybrid Search
- Dense Retrieval
- Sparse Retrieval
- BM25
- Semantic Search
- Retrieval-Augmented Generation concepts

### Embeddings

- HuggingFace
- Sentence Transformers
- `all-MiniLM-L6-v2`

### Vector Database

- Pinecone

### Frameworks & Libraries

- LangChain
- LangChain Community
- LangChain HuggingFace
- Pinecone Text
- Sentence Transformers
- Scikit-learn

### Application

- Streamlit

### Development

- Git
- GitHub
- Python Virtual Environment
- Streamlit Cloud

## 📂 Project Structure

`app.py`

Main Streamlit application containing document ingestion, embedding generation, BM25 processing, Pinecone integration, hybrid querying, and result display.

`documents.json`

Persistent local document corpus used to rebuild the BM25 representation.

`bm25_values.json`

Persisted BM25 encoder values.

`requirements.txt`

Python dependencies required to run the application.

`.gitignore`

Prevents environment files, virtual environments, caches, and other local files from being committed.

`.env`

Local environment variables containing API credentials. This file is intentionally excluded from Git.

## 🔑 Environment Variables

Create a `.env` file locally with:

PINECONE_API_KEY=your_pinecone_api_key

HF_TOKEN=your_huggingface_token

Do not commit `.env` or expose API credentials publicly.

For Streamlit Cloud, configure the same variables through the application's Secrets settings.

## ⚙️ Local Setup

### 1. Clone the repository

`git clone https://github.com/naman-DA/Hybrid-Search-RAG.git`

### 2. Move into the project directory

`cd Hybrid-Search-RAG`

### 3. Create a virtual environment

`python -m venv venv`

### 4. Activate the environment

Windows:

`venv\Scripts\activate`

### 5. Install dependencies

`pip install -r requirements.txt`

### 6. Configure environment variables

Create `.env` in the project root and add:

PINECONE_API_KEY=your_pinecone_api_key

HF_TOKEN=your_huggingface_token

### 7. Run the application

`streamlit run app.py`

The application will open in your browser.

## 🔎 Search Process

When a user enters a query, the application:

1. Generates a dense embedding for the query.
2. Encodes the query using BM25.
3. Applies the configured dense/sparse weighting.
4. Sends the combined representation to Pinecone.
5. Retrieves the top relevant documents.
6. Returns document metadata and hybrid relevance scores.
7. Displays the results in the Streamlit interface.

## 📊 Example Result

The application displays results in the following form:

Hybrid Score: 0.xxxx

Document ID: doc_x

Source: user_input

Relevant document text...

The hybrid score helps visualize the relative relevance of each retrieved document.

## 🎯 Why Hybrid Search?

Dense and sparse retrieval solve different problems.

Dense retrieval is strong at understanding semantic meaning.

For example, a query such as:

"How can I improve customer satisfaction?"

can retrieve content related to customer experience even when the exact phrase does not appear.

BM25 is strong when exact terminology matters.

For example, queries containing:

- Product names
- Technical terms
- IDs
- Specific keywords

can benefit from lexical matching.

Combining both retrieval methods provides a more flexible search system than relying on either approach independently.

## 🧪 Current Implementation

The current project focuses on the retrieval layer.

It demonstrates:

- Dense embeddings
- BM25 sparse retrieval
- Hybrid search
- Pinecone indexing
- Metadata-aware retrieval
- Relevance score visualization

The application currently returns retrieved documents rather than generating final LLM answers from the retrieved context.

## 🚀 Deployment

The application is deployed using Streamlit Community Cloud.

Live application:

https://hybrid-search-rag-wxfwxnbbmzpezg6ylwas7m.streamlit.app/

The deployment uses Streamlit Secrets for API credentials instead of committing credentials to the repository.

## 🔐 Security

Sensitive credentials are not stored in the repository.

The project excludes:

- `.env`
- `.env.*`
- Streamlit secrets
- Virtual environments
- Python cache files

API credentials should always be provided through environment variables or Streamlit Secrets.

## 📈 Future Improvements

Possible extensions include:

- Add an LLM generation layer on top of retrieved context.
- Convert the current retrieval pipeline into a complete RAG question-answering system.
- Add document upload through the Streamlit UI.
- Add document deletion and update functionality.
- Experiment with different dense/sparse weighting values.
- Add retrieval evaluation metrics such as Precision@K and Recall@K.
- Add reranking using a cross-encoder.
- Add query expansion.
- Add filtering using document metadata.
- Add conversational history for multi-turn RAG.
- Add automated evaluation datasets for retrieval quality.

## 💡 Key Learning Outcomes

This project demonstrates practical understanding of:

- Vector embeddings
- Semantic search
- Sparse retrieval
- BM25
- Hybrid retrieval
- Vector databases
- Pinecone
- HuggingFace embeddings
- LangChain integrations
- Metadata handling
- Retrieval scoring
- Environment and secret management
- Streamlit deployment

## 👨‍💻 Author

Naman Garg

B.Tech Computer Science

GitHub: https://github.com/naman-DA

LinkedIn: https://www.linkedin.com/in/naman-garg-16672b327

## ⭐ Project

If you find this project useful, consider giving the repository a star.

GitHub Repository:

https://github.com/naman-DA/Hybrid-Search-RAG