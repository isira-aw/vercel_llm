import os
import time
from typing import List
from pinecone import Pinecone, ServerlessSpec
from semantic_router.encoders import HuggingFaceEncoder
from groq import Groq

# Load API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Check or create index
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name not in existing_indexes:
    pc.create_index(name=pinecone_index_name, dimension=768, metric="cosine", spec=spec)
    while not pc.describe_index(pinecone_index_name).status.get("ready", False):
        time.sleep(1)

index = pc.Index(pinecone_index_name)
encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

# Initialize Groq
groq_client = Groq(api_key=groq_api_key)

def get_docs(query: str, top_k: int = 5) -> List[dict]:
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])
    return [match["metadata"] for match in matches] if matches else []

def generate_answer(query: str, docs: List[dict]) -> str:
    if not docs:
        return "I'm sorry, I couldn't find relevant information. Please consult a doctor."

    context_texts = [doc.get("text", "") for doc in docs]
    context = "\n---\n".join(context_texts)

    messages = [
        {"role": "system", "content": f"You are a supportive medical chatbot. CONTEXT:\n{context}"},
        {"role": "user", "content": query}
    ]

    try:
        chat_response = groq_client.chat.completions.create(model="llama3-70b-8192", messages=messages)
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
