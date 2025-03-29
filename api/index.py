from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModel
import torch
import requests
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Retrieve API Keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mommycareknowledgebase")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Pinecone initialization error: {str(e)}")
    index = None

# Initialize sentence transformer for encoding - a lightweight model for embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
try:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    print(f"Embedding model {EMBEDDING_MODEL_NAME} initialized successfully")
    
    # Mean Pooling function to get sentence embeddings
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    # Get embeddings function
    def get_embeddings(text):
        # Tokenize sentences
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        # Perform pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings[0].tolist()  # Convert to list for Pinecone
    
except Exception as e:
    print(f"Embedding model initialization error: {str(e)}")
    
    def get_embeddings(text):
        # Fallback to random vectors if model fails
        print("WARNING: Using fallback random embeddings")
        return [0.1] * 384  # MiniLM output dimension

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

def get_docs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents from Pinecone
    """
    if not index:
        print("Pinecone index not available")
        return []
        
    try:
        # Generate embeddings for the query
        query_vector = get_embeddings(query)
        
        # Query Pinecone
        res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        matches = res.get("matches", [])
        
        # Extract and return metadata
        docs = []
        for match in matches:
            if match.get("score", 0) > 0.7:  # Only include relevant matches
                metadata = match.get("metadata", {})
                if metadata:
                    docs.append(metadata)
        
        return docs
    except Exception as e:
        print(f"Error in get_docs: {str(e)}")
        return []

def generate_answer(query: str, docs: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generate answer using Hugging Face model
    """
    try:
        # Prepare context from documents if available
        context = ""
        if docs and len(docs) > 0:
            context = "Context information:\n"
            for i, doc in enumerate(docs, 1):
                text = doc.get("text", "")
                source = doc.get("source", "Unknown source")
                if text:
                    context += f"{i}. {text} (Source: {source})\n"
        
        # Prepare prompt with or without context
        if context:
            prompt = f"""<s>[INST] <<SYS>>
You are a compassionate and helpful medical chatbot designed for mothers. 
Answer questions in a friendly and supportive manner.
Base your answers on the provided context when relevant.
<<SYS>>

{context}

User question: {query} [/INST]"""
        else:
            prompt = f"""<s>[INST] <<SYS>>
You are a compassionate and helpful medical chatbot designed for mothers. 
Answer questions in a friendly and supportive manner.
<<SYS>>

User question: {query} [/INST]"""
            
        # Call Hugging Face API
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.7}}
        
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            # Extract answer from response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").replace(prompt, "").strip()
            return result.get("generated_text", "").replace(prompt, "").strip()
        else:
            # Fallback response on API error
            print(f"Hugging Face API error: {response.status_code} - {response.text}")
            return "I'm sorry, I couldn't generate an answer at this time. Please try again later."
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
        "embedding_model": "initialized" if 'model' in globals() else "not initialized",
        "huggingface_api": "configured" if HF_API_KEY else "not configured"
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
        
        # Translate if needed (detect if query is Sinhala)
        english_query = sinhala_to_english(request.query)
        print(f"Translated/processed query: {english_query}")
        
        # Retrieve relevant documents
        docs = get_docs(english_query, top_k=3)
        print(f"Retrieved {len(docs)} documents")
        
        # Generate answer
        answer = generate_answer(english_query, docs)
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
        
        # Retrieve relevant documents
        docs = get_docs(english_query, top_k=3)
        print(f"Retrieved {len(docs)} documents")
        
        # Generate answer in English
        english_answer = generate_answer(english_query, docs)
        print(f"Generated English answer: {english_answer[:100]}...")
        
        # Translate back to Sinhala
        sinhala_answer = english_to_sinhala(english_answer)
        print(f"Translated to Sinhala: length {len(sinhala_answer)}")
        
        return {"answer": sinhala_answer}
    except Exception as e:
        print(f"Error in get_answer_sinhala: {str(e)}")
        error_msg = f"I'm sorry, I couldn't process your request. Error: {str(e)}"
        return {"answer": english_to_sinhala(error_msg)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)