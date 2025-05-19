from typing import Dict, Any, List, Callable, Type, Optional
import logging
import importlib
import os
from pydantic import BaseModel
from langchain_core.tools import Tool as LangchainTool # alias to avoid confusion

logger = logging.getLogger(__name__)

_REGISTERED_TOOLS_DATA: Dict[str, Dict[str, Any]] = {}

def register_tool(name: str, description: str, args_schema: Optional[Type[BaseModel]] = None):
    """
    Decorator to register a Python function as a tool.
    Args:
        name: The unique name of the tool.
        description: A description of what the tool does, for the LLM.
        args_schema: Optional Pydantic BaseModel class defining the arguments the LLM should provide for this tool. If None, the tool is assumed to take no arguments from the LLM.
    """
    def decorator(func: Callable) -> Callable:
        if name in _REGISTERED_TOOLS_DATA:
            logger.warning(f"Tool with name '{name}' is being re-registered. Previous: {_REGISTERED_TOOLS_DATA[name]['function']}, New: {func}")
        _REGISTERED_TOOLS_DATA[name] = {
            "name": name,
            "description": description,
            "function": func,  # The actual Python function to call
            "args_schema": args_schema  # Schema for LLM to fill
        }
        logger.debug(f"Registered tool: '{name}' with schema: {args_schema}")
        return func
    return decorator

def import_tool_modules():
    """Dynamically imports all Python modules in the 'tools' directory (except __init__.py) to ensure their @register_tool decorators are executed."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name_only = filename[:-3]
            module_qualname = f"tools.{module_name_only}"
            try:
                # Check if module is already imported to prevent re-import issues in some environments
                # if module_qualname not in sys.modules:
                importlib.import_module(module_qualname)
                logger.debug(f"Successfully imported tool module: {module_qualname}")
            except Exception as e:
                logger.error(f"Error importing tool module {module_qualname}: {e}", exc_info=True)

# Import all tool modules when __init__.py is loaded
import_tool_modules()

def get_tools_for_llm_binding() -> List[LangchainTool]:
    """
    Returns a list of LangchainTool objects suitable for binding to an LLM.
    The 'args_schema' of these tools dictates what parameters the LLM will attempt to provide.
    Contextual parameters (like user_id, channel_id) are NOT part of this schema; they are injected later during actual tool execution.
    """
    llm_tools: List[LangchainTool] = []
    for name, tool_data in _REGISTERED_TOOLS_DATA.items():
        lc_tool = LangchainTool(
            name=name,
            description=tool_data["description"],
            func=tool_data["function"],
            args_schema=tool_data["args_schema"] 
        )
        llm_tools.append(lc_tool)
    logger.debug(f"Prepared {len(llm_tools)} tools for LLM binding.")
    return llm_tools

def get_tool_function_map() -> Dict[str, Callable]:
    """
    Returns a map of tool names to their actual callable Python functions.
    This map is used by the custom_tool_executor_node to execute the correct function
    associated with a tool name.
    """
    return {
        name: data["function"] for name, data in _REGISTERED_TOOLS_DATA.items()
    }

