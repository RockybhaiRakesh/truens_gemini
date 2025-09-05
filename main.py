# ---------------- Install dependencies ----------------
# pip install chromadb google-generative-ai trulens-core trulens-providers-google numpy

import os
import time
import chromadb
import google.generativeai as genai
from trulens.core import TruSession, Feedback
from trulens.apps.app import TruApp
from trulens.providers.google import Google
import numpy as np
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

# ---------------- API Keys ----------------
GEMINI_API_KEY = "AIzaSyCbwJW0un35h7rj4WxzRqVW_I-mxz_YZHw"
CHROMA_API_KEY = "ck-3gyqeVuTuHyZ1vaGibCwsW8avmjZTuxCjz5VWp1pBTYz"
CHROMA_TENANT = "6ffafc43-ad55-4687-98a1-a85df6d12130"
CHROMA_DATABASE = "gemini"

# ---------------- Configure Gemini ----------------
genai.configure(api_key=GEMINI_API_KEY)

# ---------------- Initialize Chroma Cloud ----------------
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

# ---------------- Custom embedding function ----------------
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error for '{text}': {e}")
        return None

collection = chroma_client.get_or_create_collection(
    name="demo_collection",
    embedding_function=None
)

# ---------------- Insert sample documents ----------------
documents = {
    "uw_info": "The University of Washington, founded in 1861 in Seattle, is a public research university with over 45,000 students.",
    "wsu_info": "Washington State University, founded in 1890, is a public research university in Pullman, Washington.",
    "starbucks_info": "Starbucks is an American multinational coffeehouse chain headquartered in Seattle, Washington."
}

print("Adding documents to ChromaDB...")
for doc_id, text in documents.items():
    vector = get_embedding(text)
    if vector:
        collection.add(documents=[text], ids=[doc_id], embeddings=[vector])
        print(f"Added document: {doc_id}")
    else:
        print(f"Failed to add document: {doc_id}")
    time.sleep(0.5)
print("Documents added successfully!")

# ---------------- Initialize Gemini Client ----------------
provider = Google(model_engine="gemini-1.5-flash")

# ---------------- TruLens Session ----------------
session = TruSession()
session.reset_database()

# ---------------- Define RAG Class ----------------
class RAG:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.gemini_model = genai.GenerativeModel(self.model_name)

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        vector = get_embedding(query)
        if not vector:
            return []
        results = collection.query(query_embeddings=[vector], n_results=3)
        return [doc for sublist in results["documents"] for doc in sublist]

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_completion(self, query: str, context: list) -> str:
        if not context:
            return "Sorry, I could not find an answer."
        prompt = f"Context: {context}\nAnswer the question: {query}"
        response = self.gemini_model.generate_content(prompt)
        return response.text if response.text else "No answer found."

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        context = self.retrieve(query)
        return self.generate_completion(query, context)

# ---------------- Feedback Functions ----------------
# ----------- Feedback Functions -----------
import numpy as np

f_groundedness = Feedback(
    provider.groundedness_measure_with_cot_reasons, name="Groundedness"
).on_context(collect_list=True).on_output()

f_answer_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input().on_output()

f_context_relevance = Feedback(
    provider.context_relevance_with_cot_reasons, name="Context Relevance"
).on_input().on_context(collect_list=False).aggregate(np.mean)

# ---------------- Create TruApp ----------------
rag = RAG(model_name="gemini-1.5-flash")
tru_app = TruApp(
    rag,
    app_name="GeminiChromaRAG",
    app_version="v1",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# ---------------- Run Queries ----------------
queries = [
    "What is the University of Washington?",
    "Where is Starbucks headquartered?",
    "Tell me about Washington State University."
]

if __name__ == "__main__":
    print("Running RAG application with Gemini and TruLens...")
    with tru_app as recording:
        for q in queries:
            answer = rag.query(q)
            print(f"Query: {q}\nAnswer: {answer}\n")

    # ---------------- Run TruLens Dashboard ----------------
    from trulens.dashboard import run_dashboard
    run_dashboard(session)
