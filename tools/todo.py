import json
import os
from typing import List, Dict, Any, Union, Optional
import logging
from tools import register_tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

def load_user_todos(user_id: str) -> List[Dict[str, Any]]:
    filepath = f"data/todo_{user_id}.json"
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_user_todos(user_id: str, todos: List[Dict[str, Any]]):
    filepath = f"data/todo_{user_id}.json"
    with open(filepath, 'w') as f:
        json.dump(todos, f, indent=4)

class AddTodoItemArgs(BaseModel):
    item: str = Field(description="The TODO item's content.")
        
@register_tool(
    name="add_todo_item",
    description="Adds an item to the user's TODO list.",
    args_schema=AddTodoItemArgs
)

def add_todo_item(user_id: str, item: str) -> str:
    todos = load_user_todos(user_id)
    if any(todo['description'].lower() == item.lower() and not todo['completed'] for todo in todos):
        return f"It looks like '{item}' is already on your active TODO list."
    new_item = {"id": len(todos) + 1, "description": item, "completed": False}
    todos.append(new_item)
    save_user_todos(user_id, todos)
    return f"Added '{item}' to your TODO list."


class ViewTodoListArgs(BaseModel):
    status: Optional[str] = Field(description="Return items with status. Can be 'all', 'completed' or 'pending'.",
                                  default="all")

@register_tool(
    name="view_todo_list",
    description="Shows the user their current TODO list, optionally filtering by their status.",
    args_schema=ViewTodoListArgs
)

def view_todo_list(user_id: str, status: str = "all") -> Union[str, List[str]]:
    """
    Args:
        user_id: The user's ID.
        status: 'pending', 'completed', or 'all'. Defaults to 'all'.
    """
    todos = load_user_todos(user_id)
    if not todos:
        return "Your TODO list is empty!"
    filtered_todos = []
    if status.lower() == "pending":
        filtered_todos = [item for item in todos if not item['completed']]
        if not filtered_todos:
            return "You have no pending TODO items. Great job!"
        title = "Your Pending TODO Items:"
    elif status.lower() == "completed":
        filtered_todos = [item for item in todos if item['completed']]
        if not filtered_todos:
            return "You have not completed any TODO items yet."
        title = "Your Completed TODO Items:"
    else: # all
        filtered_todos = todos
        title = "Your TODO List (All Items):"
    if not filtered_todos and status.lower() != "pending":
        return "Your TODO list is empty!"
    response_lines = [title]
    for item in filtered_todos:
        prefix = "DONE" if item['completed'] else "TODO"
        response_lines.append(f"{prefix} (ID: {item['id']}) {item['description']}")
    
    return "\n".join(response_lines)

class MarkTodoCompletedArgs(BaseModel):
    item_id: int = Field(description="The TODO item's item_id.")

@register_tool(
    name="mark_todo_completed",
    description="Marks a TODO item as completed using its ID.",
    args_schema=MarkTodoCompletedArgs
)

def mark_todo_completed(user_id: str, item_id: int) -> str:
    """Mark TODO item as completed based on its ID."""
    todos = load_user_todos(user_id)
    item_found = False
    for item in todos:
        if item['id'] == item_id:
            if item['completed']:
                return f"Item '{item['description']}' (ID: {item_id}) is already marked as completed."
            item['completed'] = True
            item_found = True
            save_user_todos(user_id, todos)
            return f"Marked '{item['description']}' (ID: {item_id}) as completed."
    if not item_found:
        return f"Could not find TODO item with ID {item_id}."
    return "Error processing request."
