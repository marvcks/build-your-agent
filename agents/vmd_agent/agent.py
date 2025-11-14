import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from dp.agent.adapter.adk import CalculationMCPToolset

# from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams


model = LiteLlm(
    # model="openai/ecnu-max",
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

BOHRIUM_STORAGE = {
    "type": "https",
    "plugin": {
        "type": "bohrium",
        "access_key": os.getenv("BOHRIUM_ACCESS_KEY"),
        "project_id": os.getenv("BOHRIUM_PROJECT_ID"),
        "app_key": "agent"
        }
    }

BOHRIUM_EXECUTOR = {
    "type": "dispatcher",
    "machine": {
        "batch_type": "OpenAPI",
        "context_type": "OpenAPI",
        "remote_profile": {
        "access_key": "a03ba3fc99c94b32afad3a94ffdcd995",
        "project_id": 596675,
        "app_key": "agent",
        "image_address": "registry.dp.tech/dptech/dp/native/prod-13364/molpilot-server:0.3.1",
        "platform": "ali",
        "machine_type": "c32_m128_cpu"
        }
    },
    "resources": {
        "envs": {
        "BOHRIUM_PROJECT_ID": 596675
        }
    }
    }


orca_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        # url="https://2f891485332423c8715842537cf742a0.app-space.dplink.cc/sse?token=82274530fa09446cbbda56c1e3afcd41"
        url="http://jyze1397882.bohrium.tech:50001/sse"
        ),
    executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=['run_orca_calculation']
    )


vmd_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url="https://2f891485332423c8715842537cf742a0.app-space.dplink.cc/sse?token=82274530fa09446cbbda56c1e3afcd41"
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=['smiles_to_xyz', 'get_vmd_manual', 'run_vmd', 'draw_esp', 'draw_orbital', 'run_multiwfn_esp', 'write_xyz_file']
    )


root_agent = LlmAgent(
# vmd_agent = LlmAgent(
    model=model, 
    name="VMD_Agent",
    description="分子结构可视化助手, 主要是用VMD软件来可视化分子结构",
    instruction=f"""
    你是一个分子结构可视化助手, 主要是用VMD软件和Multiwfn软件来可视化分子结构。

    如果用户没有给出分子的xyz文件, 你需要先调用`smiles_to_xyz`工具来将smiles字符串转换为xyz文件, 运行时使用1核心。
    如果用户给出了xyz的文件内容, 请你使用`write_xyz_file`写入文件再计算。
    如果用户需要用orca软件来计算分子的能量, 你需要先调用`run_orca_calculation`工具来计算分子的能量。

    使用`run_multiwfn_esp`工具来获得cub文件.

    你必须首先是用`get_vmd_manual`工具来获取编写tcl命令的参考资料, 编写tcl命令以调用vmd软件。
    然后运行`run_vmd`工具来执行编写的tcl命令。

    如果用户需要画ESP图, 你需要先调用`draw_esp`工具来绘制ESP图。
    如果用户需要画分子轨道图, 你需要先调用`draw_orbital`工具来绘制分子轨道图。

    所有画好图之后, 请你使用Markdown格式来展示图。
    """,
    tools=[orca_tool, vmd_tool],
    # tools=[vmd_tool, multiwfn_tool, orca_tool],
    )

