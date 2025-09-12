"""
----------------------------------------------------------------
Agent controller implementing the "plan – reason – respond" paradigm.
----------------------------------------------------------------
"""

"""
-----------------------------------------------------------------
# MODULES AND IMPORTS
-----------------------------------------------------------------
#
# In this step we import the necessary modules and services.
#
# - logging is used for logging messages
# - List is used for type hinting lists
# - Message and ChatCompletionRequest are data models for chat 
#   messages and requests
# - azure_openai_provider is the provider for interacting with 
#   Azure OpenAI
#
-----------------------------------------------------------------
"""

import logging
from typing import List
from app.models.chat import Message, ChatCompletionRequest
from app.services.llms.azure_openai_provider import AzureOpenAIProvider
from app.prompts.prompt_manager import PromptManager

"""
-----------------------------------------------------------------
# LOGGER SETUP
# ---------------------------------------------------------------
# 
# In this step we set up the logger for the module.
#
# ---------------------------------------------------------------
"""

logger = logging.getLogger(__name__)


class AgentController:
    """
    -----------------------------------------------------------------
    Agent that follows the plan-reason-respond pattern.
    -----------------------------------------------------------------
    # 
    # This class implements an agent controller that manages the
    # execution of an agent using the "plan – reason – respond"
    # paradigm. It handles the planning phase, reasoning phase, and
    # response generation phase, allowing the agent to process user
    # messages and generate helpful responses.
    #
    ------------------------------------------------------------------
    # 
    # Attributes:
    # - azure_service: Instance of Azure OpenAI provider for chat
    #   completions and model interactions.
    ------------------------------------------------------------------
    # 
    # Methods:
    # - run_agent: Main method to execute the agent workflow.
    # - _simple_planner: Determines the goal based on user messages.
    # - _think_step_by_step: Generates chain-of-thought reasoning
    #   based on the goal.
    # - _generate_final_response: Creates the final response based
    #   on the goal and reasoning.
    #
    ------------------------------------------------------------------
    # 
    # Usage:
    # agent_controller = AgentController()
    # response = await agent_controller.run_agent(messages)
    #
    ------------------------------------------------------------------
    """

    def __init__(self):
        """Initialize the agent controller."""
        self.azure_service = AzureOpenAIProvider()
    
    async def run_agent(self, messages: List[Message]) -> str:
        """
        Run the agent using the plan-reason-respond paradigm.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Final agent response
        """
        logger.info("Starting agent execution with plan-reason-respond paradigm")
        
        try:
            # Step 1: Plan - Determine the goal
            logger.info("Step 1: Planning - determining goal")
            goal = await self._simple_planner(messages)
            logger.info(f"Goal determined: {goal}")
            
            # Step 2: Reason - Generate chain-of-thought
            logger.info("Step 2: Reasoning - generating chain-of-thought")
            reasoning = await self._think_step_by_step(messages, goal)
            logger.info("Chain-of-thought reasoning completed")
            
            # Step 3: Respond - Generate final response
            logger.info("Step 3: Responding - generating final response")
            final_response = await self._generate_final_response(messages, goal, reasoning)
            logger.info("Agent execution completed successfully")
            
            return final_response
            
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _simple_planner(self, messages: List[Message]) -> str:
        """
        Simple planner that determines the goal based on user messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Determined goal as a string
        """
        
        if not self.azure_service.is_available():
            # Fallback if Azure OpenAI is not available
            user_message = next((m.content for m in messages if m.role == "user"), "")
            fallback_goal = f"Help the user with: {user_message}"
            
            return fallback_goal
        
        planning_messages = [
            Message(
                role="system",
                content=PromptManager.get_prompt("planning_assistant")
            )
        ] + messages
        
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=planning_messages,
            temperature=0.3,
            max_tokens=150
        )
        
        try:
            response = await self.azure_service.generate_response(request)
            goal = response["choices"][0]["message"]["content"]
            
            return goal
        except Exception as e:
            logger.warning(f"Planning step failed, using fallback: {str(e)}")
            user_message = next((m.content for m in messages if m.role == "user"), "")
            fallback_goal = f"Help the user with: {user_message}"
            
            return fallback_goal
    
    async def _think_step_by_step(self, messages: List[Message], goal: str) -> str:
        """
        Generate chain-of-thought reasoning based on the goal.
        
        Args:
            messages: List of chat messages
            goal: The determined goal
            
        Returns:
            Chain-of-thought reasoning as a string
        """
        
        if not self.azure_service.is_available():
            # Fallback if Azure OpenAI is not available
            fallback_reasoning = f"To achieve the goal '{goal}', I need to provide a helpful and informative response."
            
            return fallback_reasoning
        
        reasoning_messages = [
            Message(
                role="system",
                content=PromptManager.get_prompt("reasoning_assistant", goal=goal)
            )
        ] + messages
        
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=reasoning_messages,
            temperature=0.5,
            max_tokens=300
        )
        
        try:
            response = await self.azure_service.generate_response(request)
            reasoning = response["choices"][0]["message"]["content"]
            
            return reasoning
        except Exception as e:
            logger.warning(f"Reasoning step failed, using fallback: {str(e)}")
            fallback_reasoning = f"To achieve the goal '{goal}', I need to provide a helpful and informative response."
            
            return fallback_reasoning
    
    async def _generate_final_response(self, messages: List[Message], goal: str, reasoning: str) -> str:
        """
        Generate the final response based on goal and reasoning.
        
        Args:
            messages: List of chat messages
            goal: The determined goal
            reasoning: The chain-of-thought reasoning
            
        Returns:
            Final response as a string
        """
        
        if not self.azure_service.is_available():
            # Fallback if Azure OpenAI is not available
            user_message = next((m.content for m in messages if m.role == "user"), "No message")
            fallback_response = f"I understand you want help with: {user_message}. However, I'm currently running in limited mode. Please ensure Azure OpenAI is configured for full functionality."
            
            return fallback_response
        
        final_messages = [
            Message(
                role="system",
                content=PromptManager.get_prompt("final_response_generator", goal=goal, reasoning=reasoning)
            )
        ] + messages
        
        request = ChatCompletionRequest(
            model="gpt4.1-chat",
            messages=final_messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        try:
            response = await self.azure_service.generate_response(request)
            final_response = response["choices"][0]["message"]["content"]
            
            return final_response
        except Exception as e:
            logger.error(f"Final response generation failed: {str(e)}")
            user_message = next((m.content for m in messages if m.role == "user"), "No message")
            error_response = f"I apologize, but I encountered an error while processing your request about: {user_message}. Please try again later."
            
            return error_response


# Global agent instance
agent_controller = AgentController()
