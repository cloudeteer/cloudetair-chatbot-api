"""
Chat completion endpoints following OpenAI API format.
"""

"""
----------------------------------------------------------------
# MODULES AND IMPORTS
----------------------------------------------------------------
# 
# In this step we import the necessary modules.
# 
# - logging is used for logging messages
# - uuid is used to generate unique identifiers
# - fastapi is the web framework for building the API
# - APIRouter is used to create a router for the chat endpoints
# - HTTPException is used to handle HTTP errors
# - Request is used to handle incoming requests
# - StreamingResponse is used to return streaming responses
#   that build the response incrementally so that you can
#   start sending data to the client before the entire response
#   is ready
# - Dict and Any are used for type hinting
# - ChatCompletionRequest, ChatCompletionResponse, ModelsResponse
#   are data models (typing) for request and response formats
# - azure_openai_provider and echo_provider are 
#   services for handling chat completions
# - agent_controller is the controller for managing agent 
#   interactions
#
----------------------------------------------------------------
"""

import logging
import uuid
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any

from app.models.chat import ChatCompletionRequest, ModelsResponse
from app.services.llms import azure_openai_provider, echo_provider
from app.logic.agent_controller import agent_controller
from app.logic.rag_agent_controller import rag_agent_controller


"""
----------------------------------------------------------------
# LOGGER SETUP
----------------------------------------------------------------
# 
# In this step we set up the logger for the module.
# 
----------------------------------------------------------------
"""

logger = logging.getLogger(__name__)


"""
----------------------------------------------------------------
# ROUTER SETUP
----------------------------------------------------------------
# 
# In this step we create an APIRouter instance for the chat
# endpoints.
#
----------------------------------------------------------------
"""

router = APIRouter()


"""
----------------------------------------------------------------
# MODEL MAPPING
----------------------------------------------------------------
#
# Map model names to their corresponding LLM providers.
# This allows for easy extensibility when adding new providers.
#
----------------------------------------------------------------
"""

MODEL_MAPPING = {
    "gpt4.1-chat": azure_openai_provider,
    "echo-model": echo_provider,
    # Special models that don't use LLM providers directly
    "ask-confluence": None,
}


"""
----------------------------------------------------------------
# CHAT COMPLETION ENDPOINTS
----------------------------------------------------------------
#
# In this section, we define the chat completion endpoints
# following the OpenAI API format.
#
# - POST /v1/chat/completions: Create a chat completion 
#   response
# - GET /v1/models: List available models
#
# Each endpoint handles chat completions for different models
# and supports both streaming and non-streaming responses.
# The endpoints also handle agent-mode interactions and update
#
----------------------------------------------------------------
"""

@router.post("/v1/chat/completions")
async def chat_completion(req: ChatCompletionRequest, request: Request):
    """
    Create a chat completion response with optional streaming.
    
    Args:
        req: Chat completion request containing model, messages, and stream option
        
    Returns:
        Chat completion response in OpenAI API format (streaming or non-streaming)
    """
    logger.info(f"Chat completion request for model: {req.model}, streaming: {req.stream}")
    
    # Get the provider for the requested model
    provider = MODEL_MAPPING.get(req.model)
    
    # Handle agent-mode models separately
    if req.model == "agent-mode":
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported for agent-mode. Please set stream=false."
            )
        
        # Use the agent controller
        logger.info("Using agent-mode for chat completion")
        
        # Generate session ID for tracing
        session_id = request.headers.get('x-session-id', str(uuid.uuid4()))
        user_id = request.headers.get('x-user-id', 'anonymous')
        
        agent_response = await agent_controller.run_agent(req.messages)
        
        return {
            "id": "chatcmpl-agent",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": agent_response
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    # Handle ask-confluence RAG model separately
    if req.model == "ask-confluence":
        # Use the RAG agent controller
        logger.info("Using ask-confluence for chat completion")
        
        # Handle streaming vs non-streaming requests
        if req.stream:
            logger.info("Streaming ask-confluence chat completion")
            return StreamingResponse(
                rag_agent_controller.run_rag_agent_streaming(req.messages),
                media_type="text/event-stream"
            )
        else:
            logger.info("Non-streaming ask-confluence chat completion")
            rag_response = await rag_agent_controller.run_rag_agent(req.messages)
            
            return {
                "id": "chatcmpl-rag",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": rag_response
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
    
    # Handle LLM providers
    if provider is None:
        # Fallback to echo provider for unknown models
        provider = echo_provider
        logger.warning(f"Unknown model '{req.model}', falling back to echo provider")
    
    # Check if provider is available
    if not provider.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"Provider for model '{req.model}' is not available. Please check configuration."
        )
    
    # Handle streaming vs non-streaming requests
    if req.stream:
        return StreamingResponse(
            provider.generate_streaming_response(req),
            media_type="text/event-stream"
        )
    else:
        return await provider.generate_response(req)


@router.get("/v1/models", response_model=ModelsResponse)
def models() -> Dict[str, Any]:
    """
    List available models.
    
    Returns:
        List of available models in OpenAI API format
    """
    logger.info("Listing available models")
    
    available_models = [
        {"id": "echo-model", "object": "model"}
    ]
    
    # Check each provider and add their models if available
    if azure_openai_provider.is_available():
        available_models.append({"id": "gpt4.1-chat", "object": "model"})
        logger.info("Azure OpenAI model available")
    

    
    # Add ask-confluence model if vector store is available
    from app.services.vector_stores.azure_search_store import azure_search_vector_store
    if azure_search_vector_store.is_available():
        available_models.append({"id": "ask-confluence", "object": "model"})
        logger.info("Ask-Confluence RAG model available")
    
    return {
        "data": available_models
    }
