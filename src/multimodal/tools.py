from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any] # JSON Schema for args

class ToolExecutor:
    """
    Manages a registry of tools and executes them based on model output.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info(f"🔧 Registered Tool: {tool.name}")
        
    def get_schema(self) -> str:
        """
        Returns the tool definitions in a format suitable for the system prompt.
        """
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        return json.dumps(schemas, indent=2)

    def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Executes a tool and returns the result as a string.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
            
        try:
            logger.info(f"⚙️ Executing {tool_name} with {args}")
            result = self.tools[tool_name].func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

# --- Built-in Tools ---

def calculator(expression: str) -> str:
    """Safely evaluates a mathematical expression."""
    try:
        # WHITELIST: Only allow math-safe characters
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression."
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str) -> str:
    """Searches the query (Mock). Real implementation would connect to Google/Bing API."""
    return f"[Mock Search Result] Found relevant information for '{query}': The answer is 42."

# Standard Factory for default tools
def get_default_executor() -> ToolExecutor:
    executor = ToolExecutor()
    
    executor.register(Tool(
        name="calculator",
        description="Calculate a mathematical expression. Input string logic.",
        func=calculator,
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The math expression to evaluate, e.g. '2 + 2'"}
            },
            "required": ["expression"]
        }
    ))
    
    executor.register(Tool(
        name="search_web",
        description="Search the internet for information.",
        func=search_web,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    ))
    
    return executor
