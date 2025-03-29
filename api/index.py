from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

# Retrieve API Keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mommycareknowledgebase")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Pinecone (without index creation to simplify deployment)
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Just try to connect to existing index
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    print(f"Pinecone initialization error: {str(e)}")
    # We'll continue without failing to allow deployment

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Groq initialization error: {str(e)}")

# Translation functions
def sinhala_to_english(query: str) -> str:
    """Translate Sinhala to English"""
    try:
        translator = GoogleTranslator(source="si", target="en")
        return translator.translate(query)
    except Exception:
        return query  # Return original on error

def english_to_sinhala(text: str) -> str:
    """Translate English to Sinhala"""
    try:
        translator = GoogleTranslator(source="en", target="si")
        return translator.translate(text)
    except Exception:
        return text  # Return original on error

def get_docs(query: str, top_k: int = 5):
    """
    Simplified document retrieval from Pinecone
    """
    try:
        # This is a placeholder for the actual implementation
        # We'll add error handling to prevent deployment issues
        vector = [0.1] * 768  # Placeholder vector
        res = index.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = res.get("matches", [])
        return [match.get("metadata", {}) for match in matches]
    except Exception as e:
        print(f"Error in get_docs: {str(e)}")
        return []

def generate_answer(query: str, docs=None):
    """
    Simplified answer generation using Groq
    """
    try:
        # Simple system message without context for initial deployment
        system_message = (
            "You are a compassionate and helpful medical chatbot designed for mothers. "
            "Answer questions in a friendly and supportive manner."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        chat_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        answer = chat_response.choices[0].message.content
        return answer
    except Exception as e:
        return f"I'm sorry, I couldn't generate an answer at this time. Error: {str(e)}"

# FastAPI Application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "API is running"}

@app.post("/api/get_answer/")
async def api_get_answer(request: QueryRequest):
    """Get answer in English"""
    try:
        # Simplified flow
        english_query = sinhala_to_english(request.query)
        docs = get_docs(english_query, top_k=3)
        answer = generate_answer(english_query, docs)
        return {"answer": answer}
    except Exception as e:
        print(f"Error in get_answer: {str(e)}")
        return {"answer": "I'm sorry, I couldn't process your request at this time."}

@app.post("/api/get_answer_sinhala/")
async def api_get_answer_sinhala(request: QueryRequest):
    """Get answer in Sinhala"""
    try:
        english_query = sinhala_to_english(request.query)
        docs = get_docs(english_query, top_k=3)
        english_answer = generate_answer(english_query, docs)
        sinhala_answer = english_to_sinhala(english_answer)
        return {"answer": sinhala_answer}
    except Exception as e:
        print(f"Error in get_answer_sinhala: {str(e)}")
        return {"answer": "I'm sorry, I couldn't process your request at this time."}