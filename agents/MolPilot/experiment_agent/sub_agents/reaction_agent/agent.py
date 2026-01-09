import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, 
    BOHRIUM_STORAGE,
    )

reaction_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=['enumerate_reactions']
    )


model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

reaction_agent = LlmAgent(
    model=model, 
    name="Reaction_Agent",
    description="化学反应智能体. 负责进行化学反应的枚举和计算.",
    instruction=f"""
    你是一个化学反应智能体, 负责进行化学反应的枚举和计算.
    当用户给定了反应物的xyz结构，你需要要调用`enumerate_reactions`工具，进行化学反应枚举.
    """,
    tools=[
        reaction_tool
        ],
    )


