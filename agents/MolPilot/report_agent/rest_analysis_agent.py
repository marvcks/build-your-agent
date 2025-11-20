import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
# from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, BOHRIUM_STORAGE, 
)
# from ..tools import adk_tavily_tool
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from ..tools import adk_tavily_tool, get_image_from_url

dataAnalysys_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        # url=os.getenv("MOLPILOT_SERVER_URL")
        url="http://0.0.0.0:50001/sse"
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

    1.如果有图片生成, 使用`get_image_from_url`下载图片, 然后`load_artifacts_tool`读取图片.
    2. 总结分析扫描结果, 使用Markdown格式, 组织图像和文本, 生成最终的报告.

    ## 特殊任务处理
    如果进行了扫描计算, 则需要根据扫描结果绘制图像.
    1. 根据之前的REST输出结果, 调用`draw_scan_result`工具, 绘制扫描结果的图像.
    2. 使用`get_image_from_url`下载图片
    3. 使用`load_artifacts_tool`读取图片
    4. 总结分析扫描结果, 使用Markdown格式, 组织图像和文本, 生成最终的报告.

    在报告中嵌入图片时, 请使用Markdown语法, 例如:
    ```markdown
    <img src="https//xxx" alt="xxx" width="50">
    ```
    """,
    tools=[dataAnalysys_tool],
    disallow_transfer_to_parent=True,
    )

