import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, BOHRIUM_STORAGE, 
)
from ..tools import get_image_from_url, get_image
from google.adk.tools.load_artifacts_tool import load_artifacts_tool



# dataAnalysys_tool = CalculationMCPToolset(
#     connection_params=SseServerParams(
#         url=os.getenv("MOLPILOT_SERVER_URL")
#         ),
#     storage=BOHRIUM_STORAGE,
#     tool_filter=[
#         "get_data_from_opt_output",
#         "get_data_from_freq_output",
#         "plot_three_ir_spectra",
#         "plot_two_ir_spectra",
#         "plot_one_ir_spectra",
#     ],
#     )


model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )
        # 1. 使用`get_image_from_url`载入Multiwfn输出文件中的图片
        # 2. 使用`get_image`获取图片内容.
        # 3. 详细解释图片中的内容
        # 4. 根据图片内容,编写详细的分析报告. 使用Markdown格式给出报告.必须插以这样的格式"<img src="https//xxx" alt="xxx" width="50">)."插入入图片.

multiwfn_analysis_agent = LlmAgent(
    model=model, 
    name="Multiwfn_Analysis_Agent",
    description="负责从Multiwfn输出文件中提取关键数据,执行必要的后处理计算,并根据用户要求撰写清晰,美观的最终报告.",
    instruction="""
        使用`get_image_from_url`下载Multiwfn输出文件中的图片.
        使用`load_artifacts`将下载的图片加载到会话中.
        解读图片中的内容,并根据内容编写详细的分析报告.
        将Multiwfn给出的图片插入到报告中.必须插以这样的格式插入图片.例如:
        ```markdown
        ![figure xxx](https://bohrium.xxxh_x.png)  # Must use bohrium url
        ```
        """,
    tools=[get_image_from_url, load_artifacts_tool],
    disallow_transfer_to_parent=True,
    )

