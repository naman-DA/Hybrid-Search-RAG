import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

## Load Environment Variables

load_dotenv()

st.set_page_config(page_title="Hybrid Search (Dense + Sparse)", layout="centered")
st.title(" Hybrid Search using Pinecone + LangChain")

## Sidebar for API Keys

st.sidebar.header("API Keys")

# pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
# hf_token = st.sidebar.text_input("HuggingFace Token", type="password")

# if not pinecone_api_key or not hf_token:
#     st.warning("Please enter Pinecone and HuggingFace API keys.")
#     st.stop()
    
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["HF_TOKEN"] = hf_token

## Initialize Pinecone

index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone()

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = "dotproduct",
        spec = ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

## Initialize Embeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

## Initialize BM25 Encoder

bm25_file = "bm25_values.json"

if os.path.exists(bm25_file):
    bm25_encoder = BM25Encoder().load(bm25_file)
else:
    bm25_encoder = BM25Encoder().default()

## Create Hybrid Retriever

retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
)

## Add Text Section

st.subheader("Add Documents")

text_input = st.text_area("Enter text (one per line)")

if st.button("Add to Index"):
    if text_input.strip():
        documents = [line.strip() for line in text_input.split("\n") if line.strip()]

        ## Fit BM25
        
        bm25_encoder.fit(documents)
        bm25_encoder.dump(bm25_file)

        retriever.add_texts(documents)

        st.success("Documents added successfully!")
    else:
        st.error("Please enter valid text.")

## Query Section

st.subheader("Search Query")

query = st.text_input("Enter your search query")

if st.button("Search"):
    if query.strip():
        results = retriever.invoke(query)

        st.success("Results:")
        for i, doc in enumerate(results, 1):
            st.write(f"**Result {i}:**")
            st.write(doc.page_content)
            st.write("---")
    else:
        st.error("Please enter a query.")