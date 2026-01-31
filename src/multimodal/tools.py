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
    """
    Search the web for information.
    
    Attempts to use available search APIs in order of preference:
    1. SerpAPI (if SERPAPI_KEY environment variable is set)
    2. DuckDuckGo search (no API key required)
    3. Wikipedia search (no API key required)
    4. Fallback to informative message
    
    Args:
        query: Search query string
        
    Returns:
        Search results as formatted string
    """
    import os
    
    # Try SerpAPI if key is available
    serpapi_key = os.environ.get("SERPAPI_KEY")
    if serpapi_key:
        try:
            import requests
            url = "https://serpapi.com/search"
            params = {
                "q": query,
                "api_key": serpapi_key,
                "engine": "google",
                "num": 3
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = data.get("organic_results", [])
                if results:
                    formatted = f"Search results for '{query}':\n\n"
                    for i, result in enumerate(results[:3], 1):
                        title = result.get("title", "No title")
                        snippet = result.get("snippet", "No description")
                        formatted += f"{i}. {title}\n   {snippet}\n\n"
                    return formatted.strip()
        except Exception as e:
            logger.warning(f"SerpAPI search failed: {e}")
    
    # Try DuckDuckGo search
    try:
        import urllib.request
        import urllib.parse
        import json
        
        # DuckDuckGo instant answer API
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        request = urllib.request.Request(url, headers={'User-Agent': 'Nexus/1.0'})
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            # Check for instant answer
            abstract = data.get("Abstract", "")
            if abstract:
                source = data.get("AbstractSource", "Unknown")
                return f"Search result for '{query}' (from {source}):\n\n{abstract}"
            
            # Check for related topics
            related = data.get("RelatedTopics", [])
            if related:
                result = related[0].get("Text", "")
                if result:
                    return f"Search result for '{query}':\n\n{result}"
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
    
    # Try Wikipedia search as final fallback
    try:
        import urllib.request
        import urllib.parse
        import json
        
        # Wikipedia search API
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&srlimit=2"
        
        request = urllib.request.Request(url, headers={'User-Agent': 'Nexus/1.0'})
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode())
            search_results = data.get("query", {}).get("search", [])
            
            if search_results:
                formatted = f"Wikipedia search results for '{query}':\n\n"
                for i, result in enumerate(search_results, 1):
                    title = result.get("title", "")
                    snippet = result.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                    formatted += f"{i}. {title}\n   {snippet}\n\n"
                return formatted.strip()
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
    
    # Final fallback
    logger.warning("All search methods failed. Returning informative message.")
    return (
        f"Search for '{query}':\n\n"
        "No search results available. To enable web search, please:\n"
        "1. Set SERPAPI_KEY environment variable for Google search, or\n"
        "2. Ensure network connectivity for DuckDuckGo/Wikipedia search\n"
        "\nAlternatively, the model will rely on its training knowledge."
    )

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
