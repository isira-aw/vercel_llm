import os
import time
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from semantic_router.encoders import HuggingFaceEncoder
from groq import Groq

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "medical-llm-index")
groq_api_key = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
spec = ServerlessSpec(cloud="aws", region="us-west-2")

existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if pinecone_index_name not in existing_indexes:
    pc.create_index(
        name=pinecone_index_name,
        dimension=768,
        metric="cosine",
        spec=spec,
        deletion_protection=False
    )
index = pc.Index(pinecone_index_name)
time.sleep(1)

encoder = HuggingFaceEncoder(name="dwzhu/e5-base-4k")

def get_docs(query: str, top_k: int = 5) -> List[dict]:
    xq = encoder([query])
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    return [match["metadata"] for match in res.get("matches", [])]

groq_client = Groq(api_key=groq_api_key)

def generate_answer(query: str, docs: List[dict]) -> str:
    if not docs:
        return "I'm sorry, I couldn't find any relevant information. Please consult your doctor."

    context_texts = [doc.get("text", "") for doc in docs]
    context = "\n---\n".join(context_texts)

    system_message = (
        "You are a compassionate and helpful medical chatbot designed for mothers. "
        "Answer questions in a friendly and supportive manner. "
        "CONTEXT:\n" + context
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    try:
        chat_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"
