import os
import json
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

# Load Environment Variables

load_dotenv()

st.set_page_config(
    page_title="Hybrid Search using Pinecone + LangChain",
    layout="centered",
)

st.title("Hybrid Search using Pinecone + LangChain")
st.caption("Dense semantic retrieval + sparse BM25 keyword retrieval")

# API Keys

def get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

pinecone_api_key = (
    get_secret("PINECONE_API_KEY")
    or os.getenv("PINECONE_API_KEY")
)

hf_token = (
    get_secret("HF_TOKEN")
    or os.getenv("HF_TOKEN")
)

if not pinecone_api_key:
    st.error("PINECONE_API_KEY is missing.")
    st.stop()

if not hf_token:
    st.error("HF_TOKEN is missing.")
    st.stop()

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["HF_TOKEN"] = hf_token

# Configuration

index_name = "hybrid-search-langchain-pinecone-v2"
documents_file = "documents.json"
bm25_file = "bm25_values.json"

# 0.5 = equal weight to dense and sparse retrieval
alpha = 0.5

# Number of results returned
top_k = 5

# Load Document Corpus

if os.path.exists(documents_file):
    with open(
        documents_file,
        "r",
        encoding="utf-8",
    ) as file:
        document_corpus = json.load(file)
else:
    document_corpus = []

# Initialize Pinecone

pc = Pinecone(api_key=pinecone_api_key)
existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

index = pc.Index(index_name)

# Initialize Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    },
)

# Initialize BM25

if document_corpus:
    bm25_encoder = BM25Encoder()
    bm25_encoder.fit(document_corpus)
    bm25_encoder.dump(bm25_file)
else:
    if os.path.exists(bm25_file):
        bm25_encoder = BM25Encoder().load(
            bm25_file
        )
    else:
        bm25_encoder = BM25Encoder().default()

# Create Hybrid Retriever

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
    top_k=top_k,
    alpha=alpha,
)

# Add Existing Documents to Index

if document_corpus:
    index_stats = index.describe_index_stats()
    total_vectors = index_stats.get(
        "total_vector_count",
        0,
    )

    if total_vectors == 0:
        metadatas = [
            {
                "document_id": f"doc_{i + 1}",
                "source": "documents.json",
                "text": document,
            }
            for i, document in enumerate(
                document_corpus
            )
        ]

        retriever.add_texts(
            document_corpus,
            metadatas=metadatas,
        )

# Add Documents

st.subheader("Add Documents")

text_input = st.text_area(
    "Enter one document per line",
    height=150,
)

if st.button("Add Documents"):
    if not text_input.strip():
        st.error("Please enter valid text.")
    else:
        new_documents = [
            line.strip()
            for line in text_input.split("\n")
            if line.strip()
        ]

        start_id = len(document_corpus) + 1

        new_metadatas = [
            {
                "document_id": f"doc_{start_id + i}",
                "source": "user_input",
                "text": document,
            }
            for i, document in enumerate(
                new_documents
            )
        ]

        # Update complete local corpus
        document_corpus.extend(
            new_documents
        )

        # Save complete corpus
        with open(
            documents_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                document_corpus,
                file,
                indent=4,
                ensure_ascii=False,
            )

        # Rebuild BM25 using complete corpus
        bm25_encoder = BM25Encoder()

        bm25_encoder.fit(
            document_corpus
        )

        bm25_encoder.dump(
            bm25_file
        )

        # Recreate retriever
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
            top_k=top_k,
            alpha=alpha,
        )

        # Add only new documents to Pinecone
        retriever.add_texts(
            new_documents,
            metadatas=new_metadatas,
        )

        st.success(
            f"{len(new_documents)} document(s) "
            "added successfully!"
        )

# Search

st.subheader("Search Query")

query = st.text_input(
    "Enter your search query",
    placeholder="Example: What is semantic search?",
)

if st.button("Search"):
    if not query.strip():
        st.error("Please enter a search query.")
    else:
        # Generate dense vector
        
        dense_vector = embeddings.embed_query(
            query
        )

        # Generate sparse BM25 vector

        sparse_vector = (
            bm25_encoder.encode_queries(
                query
            )
        )

        # Apply hybrid weighting

        hybrid_dense = [
            value * alpha
            for value in dense_vector
        ]

        hybrid_sparse = {
            "indices": sparse_vector["indices"],
            "values": [
                value * (1 - alpha)
                for value in sparse_vector["values"]
            ],
        }

        # Query Pinecone directly

        response = index.query(
            vector=hybrid_dense,
            sparse_vector=hybrid_sparse,
            top_k=top_k,
            include_metadata=True,
        )

        matches = response.get(
            "matches",
            [],
        )

        if not matches:
            st.warning(
                "No relevant documents found."
            )
        else:
            st.success(
                f"Retrieved {len(matches)} "
                "relevant documents."
            )

            # Display Results

            for i, match in enumerate(
                matches,
                1,
            ):
                metadata = match.get(
                    "metadata",
                    {},
                )

                score = match.get(
                    "score",
                    0,
                )

                document_id = metadata.get(
                    "document_id",
                    "Unknown",
                )

                source = metadata.get(
                    "source",
                    "Unknown",
                )

                text = metadata.get(
                    "text",
                    "Document text unavailable.",
                )

                st.markdown(
                    f"### Result {i}"
                )

                st.markdown(
                    f"**Hybrid Score:** `{score:.4f}`"
                )

                st.caption(
                    f"Document ID: {document_id} "
                    f"| Source: {source}"
                )

                st.write(text)

                st.divider()