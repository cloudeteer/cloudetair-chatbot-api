"""
RAG (Retrieval-Augmented Generation) Agent Controller

This module implements a RAG agent that retrieves relevant documents from the vector store
and generates responses using the retrieved context. It follows the established architectural
patterns used in the agent_controller.py module.
"""

import logging
import json
import asyncio
from typing import List, Optional, AsyncGenerator
from urllib.parse import urlparse
import os

from app.models.chat import Message, ChatCompletionRequest
from app.models.chunk import DocumentChunk
from app.services.llms.base_llm_provider import BaseLLMProvider
from app.services.vector_stores.base_vector_store_provider import BaseVectorStore
from app.services.embeddings.base_embedding_provider import BaseEmbeddingProvider
from app.services.llms.azure_openai_provider import azure_openai_provider
from app.services.vector_stores.azure_search_store import azure_search_vector_store
from app.services.embeddings.azure_openai import azure_embedding_provider

logger = logging.getLogger(__name__)


class RAGAgentController:
    """
    RAG Agent Controller implementing retrieval-augmented generation.
    
    This controller retrieves relevant documents from the Confluence vector store
    and generates responses using the retrieved context. It uses dependency injection
    for all services to ensure testability and modularity.
    
    The RAG pipeline follows these steps:
    1. Extract user query from messages
    2. Generate embedding for the query
    3. Retrieve relevant documents from vector store
    4. Generate response using LLM with retrieved context
    5. Format response with source citations
    
    Attributes:
        vector_store: Vector store for document retrieval
        embedding_provider: Embedding provider for query vectorization
        llm_provider: LLM provider for response generation
    """
    
    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        """
        Initialize the RAG agent controller.
        
        Args:
            vector_store: Vector store for document retrieval (defaults to Azure Search)
            embedding_provider: Embedding provider for query vectorization (defaults to Azure)
            llm_provider: LLM provider for response generation (defaults to Azure OpenAI)
        """
        self.vector_store = vector_store or azure_search_vector_store
        self.embedding_provider = embedding_provider or azure_embedding_provider
        self.llm_provider = llm_provider or azure_openai_provider
        
        logger.info("RAG agent controller initialized")
    
    async def run_rag_agent(self, messages: List[Message]) -> str:
        """
        Run the RAG agent workflow.
        
        This method implements the complete RAG pipeline:
        1. Extract user query from messages
        2. Retrieve relevant documents
        3. Generate response with context
        4. Format response with source citations
        
        Args:
            messages: List of chat messages
            
        Returns:
            RAG response with AI-generated answer and source links
        """
        logger.info("Starting RAG agent execution")
        
        try:
            # Step 1: Extract user query
            query = self._extract_user_query(messages)
            logger.info(f"Extracted user query: {query[:100]}...")
            
            # Step 2: Retrieve relevant documents
            logger.info("Retrieving relevant documents from vector store")
            documents = await self._retrieve_documents(query, top_k=5)
            logger.info(f"Retrieved {len(documents)} relevant documents")
            
            # Step 3: Generate response with context
            logger.info("Generating response with retrieved context")
            response = await self._generate_response_with_context(query, documents, messages)
            logger.info("RAG agent execution completed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"RAG agent execution failed: {str(e)}")
            return self._generate_fallback_response(messages, str(e))
    
    async def run_rag_agent_streaming(self, messages: List[Message]) -> AsyncGenerator[str, None]:
        """
        Run the RAG agent workflow with streaming response.
        
        This method implements the complete RAG pipeline with streaming output:
        1. Extract user query from messages
        2. Retrieve relevant documents
        3. Generate streaming response with context
        4. Format response with source citations in streaming format
        
        Args:
            messages: List of chat messages
            
        Yields:
            Server-sent events in OpenAI streaming format
        """
        logger.info("Starting RAG agent execution with streaming")
        
        try:
            # Step 1: Extract user query
            query = self._extract_user_query(messages)
            logger.info(f"Extracted user query: {query[:100]}...")
            
            # Step 2: Retrieve relevant documents
            logger.info("Retrieving relevant documents from vector store")
            documents = await self._retrieve_documents(query, top_k=5)
            logger.info(f"Retrieved {len(documents)} relevant documents")
            
            # Step 3: Generate streaming response with context
            logger.info("Generating streaming response with retrieved context")
            async for chunk in self._generate_streaming_response_with_context(query, documents, messages):
                yield chunk
            
            logger.info("RAG agent streaming execution completed successfully")
            
        except Exception as e:
            logger.error(f"RAG agent streaming execution failed: {str(e)}")
            # Send error as streaming response
            error_response = self._generate_fallback_response(messages, str(e))
            async for chunk in self._stream_text_response(error_response):
                yield chunk

    async def _generate_streaming_response_with_context(
        self,
        query: str,
        documents: List[DocumentChunk],
        messages: List[Message]
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response using LLM with retrieved document context.
        
        Args:
            query: User query string
            documents: Retrieved document chunks
            messages: Original chat messages
            
        Yields:
            Server-sent events in OpenAI streaming format
        """
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents, 1):
            # Add document content to context
            context_parts.append(f"[Document {i}]\nTitle: {doc.title or 'Untitled'}\nContent: {doc.text}")
            
            # Build source URL for Confluence documents
            source_url = self._build_confluence_url(doc)
            if source_url:
                sources.append(f"- [{doc.title or 'Confluence Page'}]({source_url})")
        
        context = "\n\n".join(context_parts)
        
        # Handle case where no documents were found
        if not documents:
            no_docs_response = self._generate_no_documents_response(query)
            async for chunk in self._stream_text_response(no_docs_response):
                yield chunk
            return
        
        # Check if LLM provider is available
        if not self.llm_provider.is_available():
            logger.warning("LLM provider not available, returning context-only streaming response")
            context_response = self._generate_context_only_response(query, documents, sources)
            async for chunk in self._stream_text_response(context_response):
                yield chunk
            return
        
        # Build system prompt with context
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided Confluence documentation context.

Instructions:
- Use the provided document context to answer the user's question
- Be accurate and specific based on the documentation
- If the context doesn't contain enough information to fully answer the question, say so
- Provide a clear and helpful response
- Do not make up information not present in the context
- Provide also the link to the source document if available

User Query: {query}

Document Context:
{context}

Please provide a comprehensive answer based on the available documentation."""

        # Create messages for LLM
        rag_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query)
        ]
        
        try:
            # Generate streaming response using LLM
            request = ChatCompletionRequest(
                model="gpt4.1-chat",  # Use the available Azure OpenAI model
                messages=rag_messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1000,
                stream=True
            )
            
            # Send initial chunk with role
            chunk = {
                "id": "chatcmpl-rag-stream",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Stream the LLM response
            ai_response_parts = []
            async for llm_chunk in self.llm_provider.generate_streaming_response(request):
                # Extract content from the LLM streaming chunk
                if llm_chunk.startswith("data: ") and not llm_chunk.startswith("data: [DONE]"):
                    try:
                        chunk_data = llm_chunk.replace("data: ", "").strip()
                        if chunk_data:
                            chunk_json = json.loads(chunk_data)
                            delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                content = delta["content"]
                                ai_response_parts.append(content)
                                # Forward the chunk to the client
                                rag_chunk = {
                                    "id": "chatcmpl-rag-stream",
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {"index": 0, "delta": {"content": content}, "finish_reason": None}
                                    ],
                                }
                                yield f"data: {json.dumps(rag_chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue
                elif llm_chunk.startswith("data: [DONE]"):
                    break
            
            # Add sources after the main response
            if sources:
                sources_text = f"\n\n**Sources:**\n" + "\n".join(sources)
                async for chunk in self._stream_text_response(sources_text, stream_id="chatcmpl-rag-stream", start_with_content=True):
                    yield chunk
            
            # Send final chunk with finish_reason
            final_chunk = {
                "id": "chatcmpl-rag-stream",
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
        except Exception as e:
            logger.error(f"LLM streaming response generation failed: {str(e)}")
            context_response = self._generate_context_only_response(query, documents, sources)
            async for chunk in self._stream_text_response(context_response, stream_id="chatcmpl-rag-stream", start_with_content=True):
                yield chunk

    async def _stream_text_response(self, text: str, stream_id: str = "chatcmpl-rag-stream", start_with_content: bool = False) -> AsyncGenerator[str, None]:
        """
        Stream a text response in OpenAI streaming format.
        
        Args:
            text: Text to stream
            stream_id: Stream ID for the chunks
            start_with_content: If True, skip the role chunk and start with content
            
        Yields:
            Server-sent events in OpenAI streaming format
        """
        # Send initial chunk with role (unless starting with content)
        if not start_with_content:
            chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Stream the text word by word
        words = text.split()
        for i, word in enumerate(words):
            content = word if i == 0 else f" {word}"
            chunk = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.05)
        
        # Send final chunk with finish_reason
        chunk = {
            "id": stream_id,
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send done signal
        yield "data: [DONE]\n\n"
    
    def _extract_user_query(self, messages: List[Message]) -> str:
        """
        Extract the user query from chat messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            User query string
        """
        # Get the last user message as the primary query
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return "No user query found"
        
        # Use the most recent user message
        query = user_messages[-1].content.strip()
        logger.debug(f"Extracted query: {query}")
        return query
    
    async def _retrieve_documents(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query: User query string
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of relevant DocumentChunk objects
        """
        try:
            # Check if vector store is available
            if not self.vector_store.is_available():
                logger.warning(f"Vector store {self.vector_store} is not available")
                return []
            
            # Check if embedding provider is available
            if not self.embedding_provider.is_available():
                logger.warning("Embedding provider is not available")
                return []
            
            # Generate query embedding
            logger.debug("Generating query embedding")
            query_vector = await self.embedding_provider.create_embedding(query, "search")
            
            # Search vector store
            logger.debug(f"Searching vector store with top_k={top_k}")
            documents = await self.vector_store.query_by_vector(query_vector, top_k)
            
            logger.info(f"Successfully retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return []
    
    async def _generate_response_with_context(
        self,
        query: str,
        documents: List[DocumentChunk],
        messages: List[Message]
    ) -> str:
        """
        Generate response using LLM with retrieved document context.
        
        Args:
            query: User query string
            documents: Retrieved document chunks
            messages: Original chat messages
            
        Returns:
            Generated response with source citations
        """
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents, 1):
            # Add document content to context
            context_parts.append(f"[Document {i}]\nTitle: {doc.title or 'Untitled'}\nContent: {doc.text}")
            
            # Build source URL for Confluence documents
            source_url = self._build_confluence_url(doc)
            if source_url:
                sources.append(f"- [{doc.title or 'Confluence Page'}]({source_url})")
        
        context = "\n\n".join(context_parts)
        
        # Handle case where no documents were found
        if not documents:
            return self._generate_no_documents_response(query)
        
        # Check if LLM provider is available
        if not self.llm_provider.is_available():
            logger.warning("LLM provider not available, returning context-only response")
            return self._generate_context_only_response(query, documents, sources)
        
        # Build system prompt with context
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided Confluence documentation context.

Instructions:
- Use the provided document context to answer the user's question
- Be accurate and specific based on the documentation
- If the context doesn't contain enough information to fully answer the question, say so
- Provide a clear and helpful response
- Do not make up information not present in the context
- Provide also the link to the source document if available

User Query: {query}

Document Context:
{context}

Please provide a comprehensive answer based on the available documentation."""

        # Create messages for LLM
        rag_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=query)
        ]
        
        try:
            # Generate response using LLM
            request = ChatCompletionRequest(
                model="gpt4.1-chat",  # Use the available Azure OpenAI model
                messages=rag_messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1000
            )
            
            response = await self.llm_provider.generate_response(request)
            ai_response = response["choices"][0]["message"]["content"]
            
            # Combine AI response with source citations
            if sources:
                final_response = f"{ai_response}\n\n**Sources:**\n" + "\n".join(sources)
            else:
                final_response = ai_response
            
            return final_response
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {str(e)}")
            return self._generate_context_only_response(query, documents, sources)
    
    def _build_confluence_url(self, doc: DocumentChunk) -> Optional[str]:
        """
        Build Confluence page URL from document metadata.
        
        Args:
            doc: Document chunk with metadata
            
        Returns:
            Confluence page URL or None if not available
        """
        try:
            # Check if we have a source URL in metadata
            if "source_url" in doc.meta:
                return doc.meta["source_url"]
            
            # Try to build URL from Confluence configuration and document ID
            # This assumes the document ID contains the page ID
            confluence_base_url = os.getenv("CONFLUENCE_BASE_URL", "https://your-company.atlassian.net/wiki")
            
            # Extract page ID from doc_id if it follows expected format
            if doc.doc_id:
                page_id = None
                
                # Handle pure numeric doc IDs first
                if doc.doc_id.isdigit():
                    page_id = doc.doc_id
                elif "confluence" in doc.doc_id.lower() and "-" in doc.doc_id:
                    # Try to extract page ID from various possible formats
                    parts = doc.doc_id.split("-")
                    for part in reversed(parts):  # Check from end, likely to contain page ID
                        if part.isdigit():
                            page_id = part
                            break
                
                if page_id:
                    return f"{confluence_base_url}/pages/viewpage.action?pageId={page_id}"
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not build Confluence URL for document {doc.doc_id}: {str(e)}")
            return None
    
    def _generate_no_documents_response(self, query: str) -> str:
        """
        Generate response when no relevant documents are found.
        
        Args:
            query: User query string
            
        Returns:
            Appropriate response for no documents found
        """
        return (
            f"I couldn't find any relevant information in our Confluence documentation "
            f"to answer your question about: {query}\n\n"
            f"This could mean:\n"
            f"- The information hasn't been documented yet\n"
            f"- The question might be outside the scope of our current documentation\n"
            f"- Try rephrasing your question with different keywords\n\n"
            f"You might want to check the Confluence space directly or reach out to the appropriate team for more information."
        )
    
    def _generate_context_only_response(
        self,
        query: str,
        documents: List[DocumentChunk],
        sources: List[str]
    ) -> str:
        """
        Generate response using only document context when LLM is unavailable.
        
        Args:
            query: User query string
            documents: Retrieved document chunks
            sources: Source citation list
            
        Returns:
            Context-based response without LLM generation
        """
        response_parts = [
            f"I found some relevant information about your query: {query}\n",
            "Here are the most relevant sections from our documentation:\n"
        ]
        
        for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 documents
            response_parts.append(
                f"**{i}. {doc.title or 'Untitled'}**\n"
                f"{doc.text[:300]}{'...' if len(doc.text) > 300 else ''}\n"
            )
        
        if sources:
            response_parts.extend(["\n**Sources:**", "\n".join(sources)])
        
        response_parts.append(
            "\n*Note: AI response generation is currently unavailable. "
            "The above information is directly from our documentation.*"
        )
        
        return "\n".join(response_parts)
    
    def _generate_fallback_response(self, messages: List[Message], error: str) -> str:
        """
        Generate fallback response when RAG pipeline fails.
        
        Args:
            messages: Original chat messages
            error: Error message
            
        Returns:
            Fallback response explaining the issue
        """
        user_query = self._extract_user_query(messages)
        
        return (
            f"I apologize, but I encountered an issue while searching for information about: {user_query}\n\n"
            f"This might be due to:\n"
            f"- Temporary service unavailability\n"
            f"- Configuration issues with the knowledge base\n"
            f"- Network connectivity problems\n\n"
            f"Please try again in a few moments, or reach out to support if the issue persists.\n\n"
            f"Technical details: {error}"
        )


# Global RAG agent instance
rag_agent_controller = RAGAgentController()