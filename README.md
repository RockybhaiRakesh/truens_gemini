Gemini + ChromaDB RAG with TruLens
Table of Contents

Overview

Features

Requirements

Installation

Configuration

Folder Structure

Usage

Queries Example

Feedback Functions

Observability & Monitoring

Best Practices

License

Overview

This project implements an enterprise-grade Retrieval-Augmented Generation (RAG) system integrating:

Google Gemini API for generative AI

Chroma Cloud as a vector database

TruLens for feedback evaluation and observability

It is designed to be modular, production-ready, and scalable, allowing developers to run RAG queries, evaluate responses, and monitor system performance.

Features

RAG Pipeline: Query → Embed → Retrieve → Generate → Feedback

Feedback Evaluation:

Groundedness

Answer Relevance

Context Relevance

Observability: Instrumentation via OpenTelemetry spans

Dashboard: TruLens dashboard for real-time evaluation

Vector Search: ChromaDB cloud-based vector embeddings

Secure API Key Usage: Supports environment variable configuration

Requirements

Python >= 3.10

Dependencies:

pip install chromadb google-generative-ai trulens-core trulens-providers-google numpy

Installation

Clone the repository:

git clone <repo_url>
cd GeminiChromaRAG


Install dependencies:

pip install -r requirements.txt


Set environment variables (recommended) or use hardcoded API keys:

export GEMINI_API_KEY="your_gemini_api_key"
export CHROMA_API_KEY="your_chroma_api_key"
export CHROMA_TENANT="your_chroma_tenant_id"
export CHROMA_DATABASE="your_chroma_database_name"

Folder Structure
GeminiChromaRAG/
│
├── main.py                # Main RAG pipeline
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .env                   # Environment variables (optional)
└── data/                  # Optional folder for documents

Configuration

ChromaDB Cloud

API Key, Tenant ID, Database Name

Gemini API

Model: gemini-1.5-flash (default)

API Key

TruLens Feedback

Groundedness

Answer Relevance

Context Relevance

Embedding Function

Uses Gemini embeddings: models/embedding-001

Usage

Add sample documents to ChromaDB

Initialize RAG pipeline with Gemini model

Configure TruLens session and feedback functions

Run queries:

queries = [
    "What is the University of Washington?",
    "Where is Starbucks headquartered?",
    "Tell me about Washington State University."
]


Launch TruLens dashboard for monitoring:

from trulens.dashboard import run_dashboard
run_dashboard(session)

Feedback Functions

Groundedness: Measures how factual the output is

Answer Relevance: Checks if the answer addresses the question

Context Relevance: Evaluates if retrieved context is relevant

Example setup:

f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness").on_context(collect_list=True).on_output()
f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance").on_input().on_output()
f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance").on_input().on_context(collect_list=False).aggregate(np.mean)

Observability & Monitoring

Uses OpenTelemetry spans for retrieval, generation, and record root events

TruLens dashboard provides:

Query logs

Feedback scores

Context usage

Aggregated evaluation metrics

Best Practices

Use environment variables for all API keys

Enable collect_list=True for context feedback to improve metrics

Use caching or rate limiting in production for embeddings and Gemini API calls

Regularly update ChromaDB and monitor vector space performance