import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, BOHRIUM_STORAGE, 
)
# from ..tools import adk_tavily_tool


dataAnalysys_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=[
        "draw_scan_result",
    ],
    )

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

rest_analysis_agent = LlmAgent(
    model=model, 
    name="Rest_Analysis_Agent",
    description="处理REST软件的输出结果,并生成报告.",
    instruction="""
    1. 根据之前的REST输出结果, 调用`draw_scan_result`工具, 绘制扫描结果的图像.
    2. 使用Markdown格式, 组织图像和文本, 生成最终的报告.
    """,
    tools=[dataAnalysys_tool],
    disallow_transfer_to_parent=True,
    )

