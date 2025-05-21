import asyncio
import inspect
import logging
import time
from typing import Any, List, Optional, Sequence, TypedDict, Dict, Tuple
from collections import defaultdict

from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel

from setup.config import config
from core.memory import add_to_memory, get_relevant_history
from core.models import get_llm
from tools import get_tool_function_map, get_tools_for_llm_binding

logger = logging.getLogger(__name__)

IDLE_TIMEOUT_SECONDS = 90 
IDLE_CHECK_INTERVAL_SECONDS = 60

def count_tokens_for_messages(messages: Sequence[BaseMessage], llm: Optional[BaseChatModel] = None) -> int:
    """
    Counts tokens for a sequence of messages.
    Uses the LLM's built-in method if available, otherwise falls back to a rough estimate.
    """
    if not messages:
        return 0

    if llm and hasattr(llm, 'get_num_tokens_from_messages'):
        try:
            return llm.get_num_tokens_from_messages(list(messages))
        except Exception as e:
            logger.warning(f"Error using llm.get_num_tokens_from_messages: {e}. Falling back to estimate.")
    
    token_estimate = 0
    for msg in messages:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            token_estimate += len(msg.content.split()) 
        elif hasattr(msg, 'content') and isinstance(msg.content, list): 
             for item_content in msg.content:
                 if isinstance(item_content, dict) and item_content.get("type") == "text":
                    token_estimate += len(item_content.get("text","").split())
    token_estimate += len(messages) * 5 
    logger.debug(f"Used fallback token estimation: {token_estimate} tokens.")
    return token_estimate


class MemoryManager:
    def __init__(self):
        self.pending_writes: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.last_activity: Dict[Tuple[str, str], float] = defaultdict(float)
        self.active_conversations_for_tokens: Dict[Tuple[str, str], Sequence[BaseMessage]] = {}
        
        self._lock = asyncio.Lock()
        self._idle_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._task_started_internally = False

    async def _ensure_idle_check_task_started(self):
        """Ensures the background idle check task is running. Starts it if not."""
        async with self._lock:
            if not self._task_started_internally or \
               (self._idle_check_task and self._idle_check_task.done()):
                
                if self._idle_check_task and self._idle_check_task.done():
                    try:
                        await self._idle_check_task
                    except Exception as e:
                        logger.error(f"MemoryManager: Previous idle check task failed: {e}", exc_info=True)

                logger.info("MemoryManager: Starting background idle check task.")
                self._shutdown_event.clear() # Reset event if restarting
                self._idle_check_task = asyncio.create_task(self._periodic_idle_check())
                self._task_started_internally = True
                await asyncio.sleep(5) # Small delay to wait for it to start

    async def add_pending_conversation(
        self,
        user_id: str,
        channel_id: str,
        user_message: str,
        bot_response: str,
        current_llm_messages: Sequence[BaseMessage],
        llm: Optional[BaseChatModel] = None,
        attachments: List[str] = [],
    ):
        """Adds a conversation turn to the pending queue and checks commit conditions."""
        if not self._task_started_internally or \
           (self._idle_check_task and self._idle_check_task.done()):
            await self._ensure_idle_check_task_started()

        async with self._lock:
            key = (user_id, channel_id)
            timestamp = time.time()

            self.pending_writes[key].append({
                "user": user_message,
                "bot": bot_response,
                "timestamp": timestamp,
                "attachments": attachments,
            })
            self.last_activity[key] = timestamp
            self.active_conversations_for_tokens[key] = current_llm_messages
            
            current_token_count = count_tokens_for_messages(current_llm_messages, llm)
            logger.info(f"MemoryManager: Added pending write for {key}. Active context token count: {current_token_count} (using LLM method: {bool(llm and hasattr(llm, 'get_num_tokens_from_messages'))}). Pending writes: {len(self.pending_writes[key])}")

            if current_token_count > config.TOKEN_THRESHOLD:
                logger.info(f"MemoryManager: Token limit ({config.TOKEN_THRESHOLD}) exceeded for {key} (count: {current_token_count}). Committing to memory.")
                await self._commit_context_to_memory(key) # Lock is already held

    async def _commit_context_to_memory(self, key: Tuple[str, str]):
        """
        Commits all pending writes for a given context (user_id, channel_id) to memory.
        Assumes the lock is already held or it's called from a context that manages it.
        """
        user_id, channel_id = key
        
        if key not in self.pending_writes or not self.pending_writes[key]:
            logger.debug(f"MemoryManager: No pending writes to commit for {key}.")
            return

        writes_to_commit = list(self.pending_writes[key])
        logger.info(f"MemoryManager: Committing {len(writes_to_commit)} exchanges for {key} to memory.")
        
        del self.pending_writes[key]
        if key in self.active_conversations_for_tokens:
            del self.active_conversations_for_tokens[key]

        for write in writes_to_commit:
            try:
                await add_to_memory(
                    user_id=user_id,
                    channel_id=channel_id,
                    user_message=write['user'],
                    bot_response=write['bot'],
                    attachments=write.get("attachments", []),
                )
                logger.debug(f"MemoryManager: Successfully committed one exchange for {key} (User: {write['user'][:30]}...).")
            except Exception as e:
                logger.error(f"MemoryManager: Error during add_to_memory for {key} (User: {write['user'][:30]}...): {e}", exc_info=True)

        logger.info(f"MemoryManager: Finished commit process for {key}. Remaining pending writes for key: {len(self.pending_writes.get(key, []))}")

    async def _periodic_idle_check(self):
        """Periodically checks for idle contexts and commits them."""
        logger.info("MemoryManager: _periodic_idle_check task started.")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=IDLE_CHECK_INTERVAL_SECONDS)
                if self._shutdown_event.is_set(): 
                    logger.info("MemoryManager: Shutdown event received in _periodic_idle_check, exiting.")
                    break 
            except asyncio.TimeoutError:
                pass 
            except asyncio.CancelledError:
                logger.info("MemoryManager: _periodic_idle_check task was cancelled.")
                break

            if self._shutdown_event.is_set():
                break

            logger.debug("MemoryManager: Performing periodic idle check...")
            async with self._lock:
                now = time.time()
                keys_to_commit_due_to_idle: List[Tuple[str, str]] = []
                
                for key in list(self.pending_writes.keys()): 
                    if not self.pending_writes[key]: 
                        continue
                    last_active_time = self.last_activity.get(key, 0)
                    if (now - last_active_time) > IDLE_TIMEOUT_SECONDS:
                        logger.info(f"MemoryManager: Idle timeout for {key} (last active: {time.ctime(last_active_time)}). Scheduling commit.")
                        keys_to_commit_due_to_idle.append(key)
                
                for key_to_commit in keys_to_commit_due_to_idle:
                    await self._commit_context_to_memory(key_to_commit) # Lock is already held
        logger.info("MemoryManager: Periodic idle check task finished.")

    async def shutdown(self):
        """
        Shuts down the MemoryManager, committing any remaining pending writes.
        It's recommended to call this during your application's graceful shutdown.
        """
        logger.info("MemoryManager: Shutdown initiated. Committing remaining pending writes...")
        self._shutdown_event.set() # Signal the idle check loop to stop

        if self._idle_check_task and not self._idle_check_task.done():
            logger.info("MemoryManager: Waiting for idle check task to complete...")
            try:
                # Give the task a moment to finish its current cycle if it was in timeout
                await asyncio.wait_for(self._idle_check_task, timeout=IDLE_CHECK_INTERVAL_SECONDS + 5)
            except asyncio.TimeoutError:
                logger.warning("MemoryManager: Timeout waiting for idle check task to finish during shutdown. Attempting to cancel.")
                self._idle_check_task.cancel()
                try:
                    await self._idle_check_task # Await cancellation
                except asyncio.CancelledError:
                    logger.info("MemoryManager: Idle check task successfully cancelled.")
            except asyncio.CancelledError: # If it was already cancelled
                 logger.info("MemoryManager: Idle check task was already cancelled during shutdown sequence.")
            except Exception as e:
                 logger.error(f"MemoryManager: Error during idle check task shutdown: {e}", exc_info=True)


        async with self._lock:
            logger.info(f"MemoryManager: Committing all remaining contexts on shutdown. Keys: {list(self.pending_writes.keys())}")
            for key in list(self.pending_writes.keys()): 
                if self.pending_writes.get(key): 
                    logger.info(f"MemoryManager: Committing writes for {key} during shutdown.")
                    await self._commit_context_to_memory(key)
        
        self._task_started_internally = False
        logger.info("MemoryManager: Shutdown complete.")

# Global instance
memory_manager = MemoryManager()

class AgentState(TypedDict):
    user_message_content: str
    user_id: str
    channel_id: str
    messages: Sequence[BaseMessage]
    current_response: Optional[str]
    attachments: List[str]

AGENT_NODE = "agent_llm_call"
TOOL_NODE = "custom_tool_executor"

def create_agent_graph():
    """Create the LangGraph agent's workflow"""
    workflow = StateGraph(AgentState)
    workflow.add_node(AGENT_NODE, agent_node_llm_call)
    workflow.add_node(TOOL_NODE, custom_tool_executor_node)
    workflow.set_entry_point(AGENT_NODE)
    workflow.add_conditional_edges(
        AGENT_NODE, should_call_tools_or_end, {TOOL_NODE: TOOL_NODE, END: END}
    )
    workflow.add_edge(TOOL_NODE, AGENT_NODE)
    return workflow.compile()

async def agent_node_llm_call(state: AgentState) -> AgentState:
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
    
    # Check if developer message needs to be added
    is_dev_message_present = any(
        isinstance(m, AIMessage) and f"(This message was sent by my developer, <@{config.DEVELOPER_USER_ID}>)" in m.content
        for m in current_messages_list
    )
    for img_url in state.get("attachments", []):
        current_messages_list.append(
            HumanMessage(content=[{
                "type": "image",
                "source_type": "url",
                "url": img_url,
            }])
        )
        logger.info(f"Added image with URL {img_url} to message list.")
    if state['user_id'] == config.DEVELOPER_USER_ID and not is_dev_message_present:
        current_messages_list.append(AIMessage(content=f"(This message was sent by my developer, <@{config.DEVELOPER_USER_ID}>)"))

    try:
        response_message = await llm_with_tools.ainvoke(current_messages_list)
    except Exception as e:
        logger.error(f"LLM invocation error: {e}", exc_info=True)
        response_message = AIMessage(content=f"Sorry, I encountered an internal error while thinking. (Details: {str(e)})")

    updated_messages = current_messages_list + [response_message]
    
    current_response_content = None
    if isinstance(response_message, AIMessage) and response_message.content:
        if not response_message.tool_calls:
            current_response_content = response_message.content
            logger.info(f"LLM raw response: \"{response_message.content[:100]}...\"")
        elif "Sorry, I encountered an internal error" in response_message.content:
             current_response_content = response_message.content
             logger.info(f"LLM raw error: \"{response_message.content[:100]}...\"")

    return {
        **state,
        "messages": updated_messages,
        "current_response": current_response_content
    }

async def custom_tool_executor_node(state: AgentState) -> AgentState:
    logger.info(f"Node: {TOOL_NODE} (User: {state['user_id']}, Channel: {state['channel_id']})")
    last_message = state['messages'][-1]
    if not (isinstance(last_message, AIMessage) and last_message.tool_calls):
        logger.warning("Tool executor node called without tool calls. Skipping.")
        return state 
    
    tool_invocations = last_message.tool_calls
    tool_messages: List[ToolMessage] = []
    tool_function_map = get_tool_function_map()

    for tool_call in tool_invocations:
        tool_name = tool_call['name']
        llm_provided_args = tool_call['args']
        logger.info(f"Executing tool: '{tool_name}' with LLM args: {llm_provided_args}")

        if tool_name not in tool_function_map:
            logger.error(f"Tool '{tool_name}' not found.")
            tool_messages.append(ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call['id']))
            continue
        
        actual_tool_func = tool_function_map[tool_name]
        final_tool_args = {**llm_provided_args}
        sig = inspect.signature(actual_tool_func)
        if 'user_id' in sig.parameters: final_tool_args['user_id'] = state['user_id']
        if 'channel_id' in sig.parameters: final_tool_args['channel_id'] = state['channel_id']
        if 'attachments' in sig.parameters: final_tool_args['attachments'] = state.get('attachments', [])
        
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
    last_message = state['messages'][-1] if state['messages'] else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Conditional Edge: AIMessage has tool calls, routing to TOOL_NODE.")
        return TOOL_NODE
    else:
        logger.info("Conditional Edge: No further tool calls, routing to END.")
        if state.get('current_response') is None:
            if isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls:
                state['current_response'] = last_message.content
                logger.info(f"Setting current_response from last AIMessage: {last_message.content[:50]}...")
            else:
                logger.warning("Routing to END, but 'current_response' is not set and last message is not a simple AIMessage.")
        return END

async def process_message_with_graph(
    agent_graph: Any, 
    user_message_content: str, 
    user_id: str, 
    channel_id: str,
    attachments: List[str] = [],
) -> str:
    raw_history = await get_relevant_history(user_message_content, user_id, channel_id)
    messages_history: List[BaseMessage] = []
    for exchange in raw_history:
        if 'user' in exchange and exchange['user']: messages_history.append(HumanMessage(content=exchange['user']))
        if 'assistant' in exchange and exchange['assistant']: messages_history.append(AIMessage(content=exchange['assistant']))
            
    initial_messages_for_llm: List[BaseMessage] = messages_history + [HumanMessage(content=user_message_content)]

    initial_state: AgentState = {
        "user_message_content": user_message_content,
        "user_id": user_id,
        "channel_id": channel_id,
        "messages": initial_messages_for_llm,
        "attachments": attachments,
        "current_response": None,
    }
    
    final_state = None
    response_to_return = "I apologize, but I encountered an unexpected issue." 
    
    llm_instance_for_counting: Optional[BaseChatModel] = None
    try:
        llm_instance_for_counting = get_llm()
    except Exception as e:
        logger.error(f"Failed to get LLM instance for token counting: {e}", exc_info=True)

    try:
        logger.info(f"Invoking agent graph for user {user_id}, channel {channel_id}, message: \"{user_message_content[:100]}...\"")
        if attachments:
            logger.info(f"The message includes the attachments: {attachments}")
        final_state = await agent_graph.ainvoke(initial_state)
        
        response_to_return = final_state.get("current_response")
        
        if not response_to_return:
            last_msg = final_state["messages"][-1] if final_state["messages"] else None
            if isinstance(last_msg, AIMessage) and last_msg.content:
                 response_to_return = last_msg.content
                 logger.warning(f"Critical: 'current_response' empty. Used content from last AIMessage: {response_to_return[:50]}...")
            else:
                logger.error("Critical: Agent graph ended, 'current_response' empty, last AIMessage no content.")
                response_to_return = "I've completed the request but have no specific text message."
        
        await memory_manager.add_pending_conversation(
            user_id=user_id,
            channel_id=channel_id,
            user_message=user_message_content,
            bot_response=response_to_return,
            current_llm_messages=final_state["messages"],
            llm=llm_instance_for_counting,
            attachments=attachments,
        )

        logger.info(f"Final response for user {user_id}, channel {channel_id}: \"{response_to_return[:100]}...\"")
        return response_to_return
        
    except Exception as e:
        logger.error(f"Error running agent graph for user {user_id}, channel {channel_id}: {e}", exc_info=True)
        if final_state: logger.debug(f"Final state before error: {final_state}")
        else: logger.debug(f"Initial state at error: {initial_state}")
        return "I apologize, but I encountered a critical error. Please try again later."
