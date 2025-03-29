# MommyCare Medical Chatbot API

This is a simplified version of the MommyCare Medical Chatbot API designed for deployment on Vercel.

## Local Development

1. Set up environment variables in a `.env` file:
```
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=mommycareknowledgebase
GROQ_API_KEY=your_groq_key
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the local development server:
```
python main.py
```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /api/get_answer/`: Get answer in English
- `POST /api/get_answer_sinhala/`: Get answer in Sinhala

## Vercel Deployment

1. Connect your GitHub repository to Vercel
2. Add the environment variables in Vercel project settings
3. Deploy

## Project Structure

```
├── api/
│   └── index.py       # FastAPI application
├── main.py            # Local development entry point
├── requirements.txt   # Dependencies
├── vercel.json        # Vercel configuration
└── README.md          # Documentation
```