import asyncio
import inspect
import logging
from typing import Any, List, Optional, Sequence, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from setup.config import config
from core.memory import add_to_memory, get_relevant_history
from core.models import get_llm
from tools import get_tool_function_map, get_tools_for_llm_binding

logger = logging.getLogger(__name__)

# State of the agent, useful for passing context
class AgentState(TypedDict):
    user_message_content: str
    user_id: str
    channel_id: str
    messages: Sequence[BaseMessage]
    current_response: Optional[str]

AGENT_NODE = "agent_llm_call"
TOOL_NODE = "custom_tool_executor"

def create_agent_graph():
    """Create the LangGraph agent's workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node(AGENT_NODE, agent_node_llm_call)
    workflow.add_node(TOOL_NODE, custom_tool_executor_node)
    workflow.set_entry_point(AGENT_NODE)
    workflow.add_conditional_edges(
        AGENT_NODE,
        should_call_tools_or_end,
        {TOOL_NODE: TOOL_NODE, END: END}
    )
    workflow.add_edge(TOOL_NODE, AGENT_NODE) # Loop back after tool execution
    return workflow.compile()

async def agent_node_llm_call(state: AgentState) -> AgentState:
    """Bind tools and invoke the LLM on the current message."""
    logger.info(f"Node: {AGENT_NODE} (User: {state['user_id']}, Channel: {state['channel_id']})")
    llm = get_llm()
    tools_for_llm = get_tools_for_llm_binding()
    llm_with_tools = llm.bind_tools(tools_for_llm)
    current_messages_list = list(state['messages'])
    if not any(isinstance(m, SystemMessage) for m in current_messages_list):
        message_content = "You are Yui, a helpful and friendly Discord bot. Respond to messages in a fun and silly manner."
        if config.DEVELOPER_USER_ID:
            message_content += f" You were created by the user <@{config.DEVELOPER_USER_ID}>."
        current_messages_list.insert(0, SystemMessage(content=message_content))
    if state['user_id'] == config.DEVELOPER_USER_ID: # Let the bot know that the developer sent the message
        current_messages_list.append(AIMessage(content="(This message was sent by my developer, <@885027267401646121>)"))
    try:
        response_message = await llm_with_tools.ainvoke(current_messages_list)
    except Exception as e:
        logger.error(f"LLM invocation error: {e}", exc_info=True)
        response_message = AIMessage(content=f"Sorry, I encountered an internal error while thinking. (Details: {str(e)})")

    updated_messages = current_messages_list + [response_message]
    
    if isinstance(response_message, AIMessage) and response_message.content:
        if not response_message.tool_calls:
            state['current_response'] = response_message.content
            logger.info(f"LLM raw response: \"{response_message.content[:100]}...\"")
        elif "Sorry, I encountered an internal error" in response_message.content:
             state['current_response'] = response_message.content
             logger.info(f"LLM raw error: \"{response_message.content[:100]}...\"")
    else:
        # Clear if tool call
        state['current_response'] = None 
        
    return {**state, "messages": updated_messages}

async def custom_tool_executor_node(state: AgentState) -> AgentState:
    """Executes tool calls and injects context (user_id, channel_id)"""
    logger.info(f"Node: {TOOL_NODE} (User: {state['user_id']}, Channel: {state['channel_id']})")
    last_message = state['messages'][-1]
    if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
        logger.warning("Tool executor node called without tool calls in the last AIMessage. Skipping.")
        return state # Should ideally not happen with correct routing
    tool_invocations = last_message.tool_calls
    tool_messages: List[ToolMessage] = []
    tool_function_map = get_tool_function_map()
    for tool_call in tool_invocations:
        tool_name = tool_call['name']
        llm_provided_args = tool_call['args']
        logger.info(f"Executing tool: '{tool_name}' with LLM args: {llm_provided_args}")
        if tool_name not in tool_function_map:
            logger.error(f"Tool '{tool_name}' not found in tool_function_map.")
            tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call['id']))
            continue
        actual_tool_func = tool_function_map[tool_name]
        final_tool_args = {**llm_provided_args}
        sig = inspect.signature(actual_tool_func)
        if 'user_id' in sig.parameters:
            final_tool_args['user_id'] = state['user_id']
        if 'channel_id' in sig.parameters:
            final_tool_args['channel_id'] = state['channel_id']
        try:
            if inspect.iscoroutinefunction(actual_tool_func):
                output = await actual_tool_func(**final_tool_args)
            else:
                loop = asyncio.get_running_loop()
                output = await loop.run_in_executor(None, lambda: actual_tool_func(**final_tool_args))
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))
            logger.info(f"Tool '{tool_name}' output: \"{str(output)[:100]}...\"")
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
            tool_messages.append(ToolMessage(content=f"Error in {tool_name}: {str(e)}", tool_call_id=tool_call['id']))
    updated_messages = state['messages'] + tool_messages
    return {**state, "messages": updated_messages}

def should_call_tools_or_end(state: AgentState) -> str:
    """Determines routing: continue with tools, or end the process."""
    last_message = state['messages'][-1] if state['messages'] else None
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Conditional Edge: AIMessage has tool calls, routing to TOOL_NODE.")
        return TOOL_NODE
    else:
        logger.info("Conditional Edge: No further tool calls, routing to END.")
        if state.get('current_response') is None:
            logger.warning("Routing to END, but 'current_response' is not set. Attempting to use last AIMessage content if available.")
            if isinstance(last_message, AIMessage) and last_message.content:
                state['current_response'] = last_message.content
            else: # Ultimate fallback
                state['current_response'] = "I'm not sure how to respond at this moment."
        return END

async def process_message_with_graph(
    agent_graph: Any, 
    user_message_content: str, 
    user_id: str, 
    channel_id: str
) -> str:
    """Processes a user message through the agent graph and returns a response."""
    
    raw_history = await get_relevant_history(user_message_content, user_id, channel_id)
    messages_history: List[BaseMessage] = []
    for exchange in raw_history:
        if 'user' in exchange and exchange['user']:
            messages_history.append(HumanMessage(content=exchange['user']))
        if 'assistant' in exchange and exchange['assistant']:
            messages_history.append(AIMessage(content=exchange['assistant']))
            
    initial_messages: List[BaseMessage] = messages_history + [HumanMessage(content=user_message_content)]

    initial_state: AgentState = {
        "user_message_content": user_message_content,
        "user_id": user_id,
        "channel_id": channel_id,
        "messages": initial_messages,
        "current_response": None,
    }
    
    final_state = None
    try:
        logger.info(f"Invoking agent graph for user {user_id}, channel {channel_id}, message: \"{user_message_content[:100]}...\"")
        final_state = await agent_graph.ainvoke(initial_state)
        
        response_to_return = final_state.get("current_response")
        
        if not response_to_return:
            logger.error("Critical: Agent graph ended but 'current_response' is empty/None in final_state.")
            response_to_return = "I seem to be at a loss for words. Please try again."
        
        await add_to_memory(
            user_id=user_id,
            channel_id=channel_id,
            user_message=user_message_content,
            bot_response=response_to_return
        )
        logger.info(f"Final response for user {user_id}, channel {channel_id}: \"{response_to_return[:100]}...\"")
        return response_to_return
        
    except Exception as e:
        logger.error(f"Error running agent graph for user {user_id}, channel {channel_id}: {e}", exc_info=True)
        if final_state:
            logger.debug(f"Final state before error: {final_state}")
        else:
            logger.debug(f"Initial state at error: {initial_state}")
        return "I apologize, but I encountered a critical error while processing your message. Please try again later."
