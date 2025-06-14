import os
import json
import logging
import sqlite3
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    RAW_TOKEN = os.getenv("AIPROXY_TOKEN")
    AIPROXY_TOKEN = f"Bearer {RAW_TOKEN}"
    AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai"
    DB_PATH = "knowledge_base.db"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"
    EMBEDDING_URL = f"{AIPROXY_URL}/v1/embeddings"
    COMPLETION_URL = f"{AIPROXY_URL}/v1/chat/completions"
    TOP_K_RESULTS = 5
    MAX_FALLBACK_LINKS = 2

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# Utility functions
class MathUtils:
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# API service classes
class EmbeddingService:
    @staticmethod
    async def get_embedding(text: str) -> List[float]:
        """Get embedding for given text using OpenAI API."""
        headers = {
            "Authorization": Config.AIPROXY_TOKEN,
            "Content-Type": "application/json"
        }
        payload = {
            "model": Config.EMBEDDING_MODEL,
            "input": text
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(Config.EMBEDDING_URL, headers=headers, json=payload) as response:
                data = await response.json()
                logging.info(f"Embedding API response: {data}")
                
                if "data" not in data:
                    raise ValueError(f"Embedding API failed: {data}")
                
                return data["data"][0]["embedding"]

class DatabaseService:
    @staticmethod
    async def find_similar_content(
        query_embedding: List[float], 
        conn: sqlite3.Connection, 
        top_k: int = Config.TOP_K_RESULTS
    ) -> List[Tuple[float, str, str]]:
        """Find most similar content chunks from database."""
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        scored_chunks = []
        for row in rows:
            _, url, content, embedding_json, _ = row
            embedding = json.loads(embedding_json)
            similarity = MathUtils.cosine_similarity(query_embedding, embedding)
            scored_chunks.append((similarity, url, content))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return scored_chunks[:top_k]

class ChatService:
    @staticmethod
    async def query_openai_with_context(context: str, question: str) -> str:
        """Query OpenAI with context and question."""
        headers = {
            "Authorization": Config.AIPROXY_TOKEN,
            "Content-Type": "application/json"
        }
        
        system_prompt = (
            "You are a helpful virtual TA for the TDS course. Use only the provided context. "
            "Always respond in the following JSON format:\n\n"
            "{\n"
            "  \"answer\": \"<concise answer — ideally under 3 lines>\",\n"
            "  \"links\": [\n"
            "    {\"url\": \"<link_url>\", \"text\": \"<short description>\"},\n"
            "    ... up to 2 links\n"
            "  ]\n"
            "}\n\n"
            "Do NOT include explanations, markdown, or extra text outside this JSON format. "
            "If no links are relevant, return an empty links list. Keep the answer short and specific."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

        payload = {
            "model": Config.CHAT_MODEL,
            "messages": messages,
            "temperature": 0.2
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(Config.COMPLETION_URL, headers=headers, json=payload) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]

class ResponseProcessor:
    @staticmethod
    def clean_gpt_response(text: str) -> Dict[str, Any]:
        """Clean and parse GPT response to structured format."""
        try:
            # Remove code block markers if present
            if text.strip().startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1]).strip()
            
            # Parse JSON response
            if text.strip().startswith("{") and text.strip().endswith("}"):
                parsed = json.loads(text)
                
                # Ensure links have text
                for link in parsed.get("links", []):
                    if not link.get("text"):
                        link["text"] = link.get("url", "Reference")
                
                return {
                    "answer": parsed.get("answer", "").strip(),
                    "links": parsed.get("links", [])
                }
        except Exception as e:
            logging.warning(f"Failed to parse GPT response as JSON: {e}")

        return {
            "answer": text.strip() or "⚠️ No answer generated.",
            "links": []
        }
    
    @staticmethod
    def generate_fallback_links(chunks: List[Tuple[float, str, str]]) -> List[Dict[str, str]]:
        """Generate fallback links from chunks."""
        fallback_links = []
        seen_urls = set()
        
        for chunk in chunks:
            url = chunk[1]
            if "discourse.onlinedegree.iitm.ac.in" in url and url not in seen_urls:
                seen_urls.add(url)
                fallback_links.append({
                    "url": url, 
                    "text": "Refer to this related discussion."
                })
                if len(fallback_links) == Config.MAX_FALLBACK_LINKS:
                    break
        
        return fallback_links

# Main service orchestrator
class KnowledgeBaseService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.database_service = DatabaseService()
        self.chat_service = ChatService()
        self.response_processor = ResponseProcessor()
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process knowledge base query and return response."""
        try:
            logging.info(f"🔍 Received question: {request.question}")
            if request.image:
                logging.info(f"🖼️ Received base64 image (length {len(request.image)})")

            # Get query embedding
            query_embedding = await self.embedding_service.get_embedding(request.question)
            
            # Find similar content
            conn = sqlite3.connect(Config.DB_PATH)
            try:
                top_chunks = await self.database_service.find_similar_content(query_embedding, conn)
            finally:
                conn.close()
            
            # Prepare context
            context = "\n\n".join(chunk[2] for chunk in top_chunks)
            
            # Generate fallback links
            fallback_links = self.response_processor.generate_fallback_links(top_chunks)
            
            # Get AI response
            raw_answer = await self.chat_service.query_openai_with_context(context, request.question)
            logging.info(f"🔹 Raw LLM response: {raw_answer}")
            
            # Process response
            parsed = self.response_processor.clean_gpt_response(raw_answer)
            if not parsed.get("answer"):
                parsed["answer"] = "⚠️ No answer generated."
            
            # Use parsed links or fallback
            links = parsed.get("links", []) if parsed.get("links") else fallback_links
            
            response = {
                "answer": parsed["answer"],
                "links": links
            }
            
            logging.info(f"✅ Final API response: {json.dumps(response, indent=2)}")
            return QueryResponse(answer=response["answer"], links=response["links"])
            
        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return QueryResponse(answer="⚠️ Failed to get an answer.", links=[])

# FastAPI application setup
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="Knowledge Base API", version="1.0.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# Initialize app and service
app = create_app()
knowledge_service = KnowledgeBaseService()

# Configure logging
logging.basicConfig(level=logging.INFO)

# API endpoints
@app.post("/api/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Main API endpoint for querying the knowledge base."""
    return await knowledge_service.process_query(request)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)