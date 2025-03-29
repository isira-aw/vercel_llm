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

# Initialize clients with proper error handling
# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Pinecone initialization error: {str(e)}")
    index = None

# Initialize Groq client - this was missing/not properly defined
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized successfully")
except Exception as e:
    print(f"Groq initialization error: {str(e)}")
    # Create a placeholder function to prevent errors
    class DummyGroq:
        def __init__(self):
            self.chat = self
            self.completions = self
        
        def create(self, model, messages):
            class DummyResponse:
                def __init__(self):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': "I'm sorry, the language model is currently unavailable."
                        })
                    })]
            return DummyResponse()
    
    groq_client = DummyGroq()

# Translation functions
def sinhala_to_english(query: str) -> str:
    """Translate Sinhala to English"""
    try:
        translator = GoogleTranslator(source="si", target="en")
        return translator.translate(query)
    except Exception as e:
        print(f"Translation error (si->en): {str(e)}")
        return query  # Return original on error

def english_to_sinhala(text: str) -> str:
    """Translate English to Sinhala"""
    try:
        translator = GoogleTranslator(source="en", target="si")
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error (en->si): {str(e)}")
        return text  # Return original on error

def get_docs(query: str, top_k: int = 5):
    """
    Simplified document retrieval from Pinecone
    """
    if not index:
        print("Pinecone index not available")
        return []
        
    try:
        # Create a simple vector for testing
        # In production, you would use an encoder here
        vector = [0.1] * 768  # Placeholder vector
        res = index.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = res.get("matches", [])
        return [match.get("metadata", {}) for match in matches]
    except Exception as e:
        print(f"Error in get_docs: {str(e)}")
        return []

def generate_answer(query: str, docs=None):
    """
    Generate answer using Groq
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
        print(f"Error generating answer: {str(e)}")
        return f"I'm sorry, I couldn't generate an answer at this time. Technical issue: {str(e)}"

# FastAPI Application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    """Health check endpoint"""
    status = {
        "status": "API is running",
        "pinecone": "connected" if index else "disconnected",
        "groq": "connected" if not isinstance(groq_client, DummyGroq) else "disconnected"
    }
    return status

@app.post("/api/get_answer/")
async def api_get_answer(request: QueryRequest):
    """Get answer in English"""
    try:
        if not request.query or request.query.strip() == "":
            return {"answer": "Please provide a question."}
            
        # Log the incoming request
        print(f"Received query: {request.query}")
        
        # Translate if needed
        english_query = sinhala_to_english(request.query)
        print(f"Translated query: {english_query}")
        
        # Skip document retrieval for initial testing
        # docs = get_docs(english_query, top_k=3)
        # print(f"Retrieved {len(docs)} documents")
        
        # Generate answer directly 
        answer = generate_answer(english_query)
        print(f"Generated answer length: {len(answer)}")
        
        return {"answer": answer}
    except Exception as e:
        print(f"Error in get_answer: {str(e)}")
        return {"answer": f"I'm sorry, I couldn't process your request. Error: {str(e)}"}

@app.post("/api/get_answer_sinhala/")
async def api_get_answer_sinhala(request: QueryRequest):
    """Get answer in Sinhala"""
    try:
        if not request.query or request.query.strip() == "":
            sinhala_response = english_to_sinhala("Please provide a question.")
            return {"answer": sinhala_response}
            
        # Log the incoming request
        print(f"Received Sinhala query: {request.query}")
        
        # Translate to English
        english_query = sinhala_to_english(request.query)
        print(f"Translated to English: {english_query}")
        
        # Generate answer in English
        english_answer = generate_answer(english_query)
        print(f"Generated English answer: {english_answer[:100]}...")
        
        # Translate back to Sinhala
        sinhala_answer = english_to_sinhala(english_answer)
        print(f"Translated to Sinhala: length {len(sinhala_answer)}")
        
        return {"answer": sinhala_answer}
    except Exception as e:
        print(f"Error in get_answer_sinhala: {str(e)}")
        error_msg = f"I'm sorry, I couldn't process your request. Error: {str(e)}"
        return {"answer": english_to_sinhala(error_msg)}