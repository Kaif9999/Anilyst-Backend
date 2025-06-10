from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anilyst Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000","http://anilyst.tech","https://www.anilyst.tech"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import after app creation to avoid circular imports
try:
    from app.services.ai_data_analyst_agent import get_ai_data_analyst_agent
    logger.info("AI Data Analyst Agent imported successfully")
except ImportError as e:
    logger.error(f"Failed to import AI Data Analyst Agent: {e}")
    get_ai_data_analyst_agent = None

@app.post("/ai-chat")
async def ai_chat(request: dict):
    """Enhanced AI chat endpoint with better error handling"""
    try:
        if get_ai_data_analyst_agent is None:
            raise HTTPException(
                status_code=500, 
                detail="AI agent service is not available"
            )
        
        message = request.get("message", "")
        dataset = request.get("dataset")
        context = request.get("context", {})
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"AI Chat request: {message[:50]}...")
        
        # Call the AI agent with the correct parameters
        result = await get_ai_data_analyst_agent(
            message=message,  # Pass as keyword argument
            data=dataset,     # Pass dataset as data
            context=context   # Pass context
        )
        
        logger.info(f"Response: {result.get('status', 'unknown')}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in AI chat: {e}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_agent_available": get_ai_data_analyst_agent is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)