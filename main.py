from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from chatbot import get_docs, generate_answer
from translation import sinhalaToEnglish, englishToSinhala
import speech_recognition as sr
import os
import tempfile

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/get_answer/")
def get_answer(request: QueryRequest):
    try:
        english_query = sinhalaToEnglish(request.query)
        docs = get_docs(english_query, top_k=5)
        answer = generate_answer(english_query, docs)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer_sinhala/")
def get_answer_sinhala(request: QueryRequest):
    try:
        english_query = sinhalaToEnglish(request.query)
        docs = get_docs(english_query, top_k=5)
        english_answer = generate_answer(english_query, docs)
        sinhala_answer = englishToSinhala(english_answer)
        return {"answer": sinhala_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_answer_voice/")
async def get_answer_voice(audio_file: UploadFile = File(...)):
    temp_audio_path = None
    try:
        fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        content = await audio_file.read()
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(content)
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            text_query = recognizer.recognize_google(audio_data)
        docs = get_docs(text_query, top_k=5)
        answer = generate_answer(text_query, docs)
        return {"query": text_query, "answer": answer}
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Speech could not be understood")
    except sr.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Speech recognition service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
