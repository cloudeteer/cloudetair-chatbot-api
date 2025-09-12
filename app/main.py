"""
----------------------------------------------------------------
# Chatbot API - FastAPI Application Entry Point
----------------------------------------------------------------
# 
# A simple FastAPI application that serves as a container for 
# the development of LLM Powered Chatbots and agents.
#
# The application will be consumed by a Open WebUI Frontend.
#
----------------------------------------------------------------
"""

import os
from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.chat import router as chat_router
from app.utils.logging import setup_logging, get_logger

"""
----------------------------------------------------------------
# APPLICATION INITIALIZATION
----------------------------------------------------------------
"""

load_dotenv()
setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

app = FastAPI(
    title="Chatbot API",
    description="A simple FastAPI application for LLM Powered Chatbots and agents",
    version="1.0.0"
)

app.include_router(chat_router)

logger.info("Chatbot API initialized successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
