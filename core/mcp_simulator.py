"""
Model Context Protocol (MCP) educational simulator.
Simulates MCP Hosts, Clients, Servers, Transports, and the three primitives
(Resources, Tools, Prompts) for interactive learning.

Reference: https://modelcontextprotocol.io
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Primitives
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MCPResource:
    """A read-only data source exposed by an MCP Server."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    content: str = ""

    def read(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "mimeType": self.mime_type,
            "text": self.content,
        }


@dataclass
class MCPToolParam:
    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class MCPTool:
    """An executable action exposed by an MCP Server."""
    name: str
    description: str
    parameters: list[MCPToolParam] = field(default_factory=list)
    handler: Callable[..., str] | None = None

    def input_schema(self) -> dict:
        props = {}
        required = []
        for p in self.parameters:
            props[p.name] = {"type": p.type, "description": p.description}
            if p.required:
                required.append(p.name)
        return {"type": "object", "properties": props, "required": required}

    def execute(self, **kwargs) -> dict:
        start = time.time()
        try:
            result = self.handler(**kwargs) if self.handler else "No handler"
            return {
                "tool": self.name,
                "result": str(result),
                "success": True,
                "duration_ms": (time.time() - start) * 1000,
            }
        except Exception as e:
            return {
                "tool": self.name,
                "result": f"Error: {e}",
                "success": False,
                "duration_ms": (time.time() - start) * 1000,
            }


@dataclass
class MCPPrompt:
    """A reusable prompt template exposed by an MCP Server."""
    name: str
    description: str
    arguments: list[dict] = field(default_factory=list)
    template: str = ""

    def render(self, **kwargs) -> list[dict]:
        text = self.template
        for key, value in kwargs.items():
            text = text.replace(f"{{{key}}}", str(value))
        return [{"role": "user", "content": {"type": "text", "text": text}}]


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Transport
# ═══════════════════════════════════════════════════════════════════════════

class TransportType(Enum):
    STDIO = "stdio"
    SSE = "sse"


@dataclass
class MCPMessage:
    """A JSON-RPC 2.0 message used by MCP."""
    id: str = ""
    method: str = ""
    params: dict = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_jsonrpc(self) -> dict:
        msg = {"jsonrpc": "2.0", "id": self.id}
        if self.method:
            msg["method"] = self.method
            if self.params:
                msg["params"] = self.params
        elif self.error:
            msg["error"] = {"code": -1, "message": self.error}
        else:
            msg["result"] = self.result
        return msg


class SimulatedTransport:
    """Simulates MCP message transport for educational purposes."""

    def __init__(self, transport_type: TransportType = TransportType.STDIO):
        self.type = transport_type
        self.message_log: list[MCPMessage] = []

    def send(self, message: MCPMessage) -> MCPMessage:
        self.message_log.append(message)
        return message

    def get_log(self) -> list[dict]:
        return [m.to_jsonrpc() for m in self.message_log]


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Server
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ServerCapabilities:
    resources: bool = False
    tools: bool = False
    prompts: bool = False


class MCPServer:
    """Simulated MCP Server that exposes resources, tools, and prompts."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._resources: dict[str, MCPResource] = {}
        self._tools: dict[str, MCPTool] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self.transport = SimulatedTransport()
        self._request_log: list[dict] = []

    @property
    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            resources=bool(self._resources),
            tools=bool(self._tools),
            prompts=bool(self._prompts),
        )

    def add_resource(self, resource: MCPResource):
        self._resources[resource.uri] = resource

    def add_tool(self, tool: MCPTool):
        self._tools[tool.name] = tool

    def add_prompt(self, prompt: MCPPrompt):
        self._prompts[prompt.name] = prompt

    def handle_request(self, method: str, params: dict | None = None) -> dict:
        """Process an incoming JSON-RPC request."""
        params = params or {}
        start = time.time()

        request = MCPMessage(method=method, params=params)
        self.transport.send(request)

        result = self._dispatch(method, params)
        duration = (time.time() - start) * 1000

        response = MCPMessage(result=result)
        self.transport.send(response)

        log_entry = {
            "method": method,
            "params": params,
            "result": result,
            "duration_ms": duration,
            "request_id": request.id,
        }
        self._request_log.append(log_entry)
        return log_entry

    def _dispatch(self, method: str, params: dict) -> Any:
        handlers = {
            "initialize": self._handle_initialize,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
        }
        handler = handlers.get(method)
        if handler:
            return handler(params)
        return {"error": f"Unknown method: {method}"}

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {
                "resources": {} if self._resources else None,
                "tools": {} if self._tools else None,
                "prompts": {} if self._prompts else None,
            },
        }

    def _handle_resources_list(self, params: dict) -> dict:
        return {
            "resources": [
                {"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
                for r in self._resources.values()
            ]
        }

    def _handle_resources_read(self, params: dict) -> dict:
        uri = params.get("uri", "")
        resource = self._resources.get(uri)
        if resource:
            return {"contents": [resource.read()]}
        return {"error": f"Resource not found: {uri}"}

    def _handle_tools_list(self, params: dict) -> dict:
        return {
            "tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema()}
                for t in self._tools.values()
            ]
        }

    def _handle_tools_call(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        tool = self._tools.get(name)
        if tool:
            return tool.execute(**arguments)
        return {"error": f"Tool not found: {name}"}

    def _handle_prompts_list(self, params: dict) -> dict:
        return {
            "prompts": [
                {"name": p.name, "description": p.description, "arguments": p.arguments}
                for p in self._prompts.values()
            ]
        }

    def _handle_prompts_get(self, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        prompt = self._prompts.get(name)
        if prompt:
            return {"messages": prompt.render(**arguments)}
        return {"error": f"Prompt not found: {name}"}

    def get_request_log(self) -> list[dict]:
        return list(self._request_log)


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Client
# ═══════════════════════════════════════════════════════════════════════════

class MCPClient:
    """Simulated MCP Client that connects to a server."""

    def __init__(self):
        self.server: MCPServer | None = None
        self.server_info: dict = {}
        self.capabilities: dict = {}
        self.connected = False

    def connect(self, server: MCPServer) -> dict:
        self.server = server
        result = server.handle_request("initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "MCPClient", "version": "1.0.0"},
        })
        self.server_info = result.get("result", {}).get("serverInfo", {})
        self.capabilities = result.get("result", {}).get("capabilities", {})
        self.connected = True
        return result

    def list_resources(self) -> list[dict]:
        if not self.server:
            return []
        result = self.server.handle_request("resources/list")
        return result.get("result", {}).get("resources", [])

    def read_resource(self, uri: str) -> dict:
        if not self.server:
            return {}
        result = self.server.handle_request("resources/read", {"uri": uri})
        return result.get("result", {})

    def list_tools(self) -> list[dict]:
        if not self.server:
            return []
        result = self.server.handle_request("tools/list")
        return result.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> dict:
        if not self.server:
            return {}
        result = self.server.handle_request("tools/call", {"name": name, "arguments": arguments})
        return result.get("result", {})

    def list_prompts(self) -> list[dict]:
        if not self.server:
            return []
        result = self.server.handle_request("prompts/list")
        return result.get("result", {}).get("prompts", [])

    def get_prompt(self, name: str, arguments: dict) -> dict:
        if not self.server:
            return {}
        result = self.server.handle_request("prompts/get", {"name": name, "arguments": arguments})
        return result.get("result", {})


# ═══════════════════════════════════════════════════════════════════════════
#  MCP Host (LLM Application)
# ═══════════════════════════════════════════════════════════════════════════

class MCPHost:
    """Simulated MCP Host — the LLM application that manages clients."""

    def __init__(self, name: str = "Learning Lab Host"):
        self.name = name
        self.clients: dict[str, MCPClient] = {}

    def create_client(self, server_name: str) -> MCPClient:
        client = MCPClient()
        self.clients[server_name] = client
        return client

    def connect_server(self, server: MCPServer) -> dict:
        client = self.create_client(server.name)
        return client.connect(server)

    def get_all_tools(self) -> dict[str, list[dict]]:
        all_tools = {}
        for name, client in self.clients.items():
            if client.connected:
                all_tools[name] = client.list_tools()
        return all_tools

    def get_all_resources(self) -> dict[str, list[dict]]:
        all_resources = {}
        for name, client in self.clients.items():
            if client.connected:
                all_resources[name] = client.list_resources()
        return all_resources


# ═══════════════════════════════════════════════════════════════════════════
#  Pre-built Demo Servers
# ═══════════════════════════════════════════════════════════════════════════

def create_weather_server() -> MCPServer:
    """Demo MCP server that provides weather data."""
    server = MCPServer("weather-server", "1.0.0")

    server.add_resource(MCPResource(
        uri="weather://current/london",
        name="London Weather",
        description="Current weather conditions in London",
        content="Temperature: 15°C, Humidity: 72%, Condition: Partly Cloudy, Wind: 12 km/h NW",
    ))
    server.add_resource(MCPResource(
        uri="weather://current/tokyo",
        name="Tokyo Weather",
        description="Current weather conditions in Tokyo",
        content="Temperature: 22°C, Humidity: 65%, Condition: Sunny, Wind: 8 km/h E",
    ))
    server.add_resource(MCPResource(
        uri="weather://forecast/london",
        name="London 3-Day Forecast",
        description="3-day weather forecast for London",
        content="Day 1: 16°C, Rain\nDay 2: 14°C, Cloudy\nDay 3: 17°C, Sunny",
    ))

    server.add_tool(MCPTool(
        name="get_temperature",
        description="Get the current temperature for a city",
        parameters=[MCPToolParam("city", "string", "The city name")],
        handler=lambda city: {"london": "15°C", "tokyo": "22°C", "new york": "18°C", "paris": "20°C"}.get(city.lower(), f"No data for {city}"),
    ))
    server.add_tool(MCPTool(
        name="get_alerts",
        description="Get active weather alerts for a region",
        parameters=[MCPToolParam("region", "string", "The region to check")],
        handler=lambda region: f"No active weather alerts for {region}." if "storm" not in region.lower() else f"ALERT: Severe storm warning for {region}!",
    ))

    server.add_prompt(MCPPrompt(
        name="weather_report",
        description="Generate a weather report for a location",
        arguments=[{"name": "location", "description": "The location", "required": True}],
        template="Generate a detailed weather report for {location}. Include temperature, humidity, wind conditions, and a brief forecast.",
    ))

    return server


def create_database_server() -> MCPServer:
    """Demo MCP server that simulates database access."""
    server = MCPServer("database-server", "1.0.0")

    server.add_resource(MCPResource(
        uri="db://schema/users",
        name="Users Table Schema",
        description="Schema of the users table",
        content="CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(100), role VARCHAR(50), created_at TIMESTAMP)",
    ))
    server.add_resource(MCPResource(
        uri="db://schema/orders",
        name="Orders Table Schema",
        description="Schema of the orders table",
        content="CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, product VARCHAR(100), amount DECIMAL, status VARCHAR(20), created_at TIMESTAMP)",
    ))

    _fake_db = {
        "users": [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
            {"id": 3, "name": "Carol", "email": "carol@example.com", "role": "user"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "product": "Widget A", "amount": 29.99, "status": "shipped"},
            {"id": 102, "user_id": 2, "product": "Widget B", "amount": 49.99, "status": "pending"},
            {"id": 103, "user_id": 1, "product": "Widget C", "amount": 19.99, "status": "delivered"},
        ],
    }

    def _query(sql: str) -> str:
        sql_lower = sql.lower().strip()
        if "users" in sql_lower and "count" in sql_lower:
            return json.dumps({"count": len(_fake_db["users"])})
        elif "users" in sql_lower:
            return json.dumps(_fake_db["users"], indent=2)
        elif "orders" in sql_lower and "count" in sql_lower:
            return json.dumps({"count": len(_fake_db["orders"])})
        elif "orders" in sql_lower:
            return json.dumps(_fake_db["orders"], indent=2)
        return json.dumps({"message": "Query executed", "rows_affected": 0})

    server.add_tool(MCPTool(
        name="query",
        description="Execute a SQL query against the database",
        parameters=[MCPToolParam("sql", "string", "The SQL query to execute")],
        handler=_query,
    ))

    server.add_prompt(MCPPrompt(
        name="sql_assistant",
        description="Help write SQL queries",
        arguments=[
            {"name": "question", "description": "What data do you need?", "required": True},
            {"name": "table", "description": "Which table to query", "required": False},
        ],
        template="You are a SQL assistant. The user needs help querying a database.\nQuestion: {question}\nTable: {table}\nWrite the SQL query and explain what it does.",
    ))

    return server


def create_filesystem_server() -> MCPServer:
    """Demo MCP server that simulates file system access."""
    server = MCPServer("filesystem-server", "1.0.0")

    _files = {
        "readme.md": "# My Project\nThis is a sample project.",
        "config.json": '{"debug": true, "port": 3000, "database": "postgres://localhost/mydb"}',
        "notes.txt": "Meeting notes from 2024-01-15:\n- Discuss Q1 roadmap\n- Review architecture\n- Plan sprint",
    }

    for fname, content in _files.items():
        server.add_resource(MCPResource(
            uri=f"file:///{fname}",
            name=fname,
            description=f"Contents of {fname}",
            mime_type="application/json" if fname.endswith(".json") else "text/plain",
            content=content,
        ))

    server.add_tool(MCPTool(
        name="read_file",
        description="Read the contents of a file",
        parameters=[MCPToolParam("path", "string", "File path to read")],
        handler=lambda path: _files.get(path.split("/")[-1], f"File not found: {path}"),
    ))
    server.add_tool(MCPTool(
        name="list_files",
        description="List files in a directory",
        parameters=[MCPToolParam("directory", "string", "Directory path", required=False)],
        handler=lambda directory="./": json.dumps(list(_files.keys())),
    ))

    return server


DEMO_SERVERS = {
    "weather": {"factory": create_weather_server, "description": "Weather data, alerts, and forecasts", "icon": "🌤️"},
    "database": {"factory": create_database_server, "description": "SQL database with users and orders tables", "icon": "🗄️"},
    "filesystem": {"factory": create_filesystem_server, "description": "File system access (read, list files)", "icon": "📁"},
}
