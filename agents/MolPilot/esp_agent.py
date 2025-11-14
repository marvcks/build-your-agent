import os
import requests

from dp.agent.adapter.adk import CalculationMCPToolset

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools import ToolContext

from google import genai
from google.genai import types


async def get_image_from_url(image_url: str, tool_context: ToolContext):

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content

        counter = str(tool_context.state.get("loop_iteration", 0))
        artifact_name = f"generated_image_" + counter + ".png"

        # Save as ADK artifact (optional, if still needed by other ADK components)
        report_artifact = types.Part.from_bytes(
            data=image_bytes, mime_type="image/png"
        )

        await tool_context.save_artifact(artifact_name, report_artifact)
        print(f"Image also saved as ADK artifact: {artifact_name}")

        return {
            "status": "success",
            "message": f"Image generated. ADK artifact: {artifact_name}.",
            "artifact_name": artifact_name,
        }
    except Exception as e:
        return {"status": "error", "message": "No images generated.  {e}"}


# not used.
async def get_image(tool_context: ToolContext):
    try:
        
        artifact_name = (
            f"generated_image_" + str(tool_context.state.get("loop_iteration", 0)) + ".png"
        )
        artifact = await tool_context.load_artifact(artifact_name)
    
        return {
            "status": "success",
            "message": f"Image artifact {artifact_name} successfully loaded."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading artifact {artifact_name}: {str(e)}"
        }


model = LiteLlm(
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
        "image_address": "registry.dp.tech/dptech/dp/native/prod-13364/molpilot-server:0.3.3",
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
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=['run_orca_calculation']
    )

vmd_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=[
        'get_vmd_manual', 'run_vmd', 'draw_esp', 'draw_orbital', 
        'run_multiwfn_esp', 'run_multiwfn_get_esp_min_max', 'run_multiwfn_draw_area', 'draw_colorbar_for_esp', 
        'run_multiwfn_orbital', 'run_multiwfn_fukui'
        ]
    )


esp_agent = LlmAgent(
    model=model, 
    name="ESP_Agent",
    description="分子ESP可视化助手, 主要是用VMD软件和Multiwfn软件来可视化分子ESP",
    instruction=f"""
    你是一个分子ESP可视化助手, 主要是用VMD软件和Multiwfn软件来可视化分子ESP。

    如果用户没有给出分子的xyz文件, 你需要先调用`smiles_to_xyz`工具来将smiles字符串转换为xyz文件, 运行时使用1核心。
    如果用户给出了xyz的文件内容, 请你使用`write_xyz_file`写入文件再计算。
    如果用户需要用orca软件来计算分子的能量, 你需要先调用`run_orca_calculation`工具来计算分子的能量。

    使用`run_multiwfn_esp`工具来获得cub文件.

    你必须首先是用`get_vmd_manual`工具来获取编写tcl命令的参考资料, 编写tcl命令以调用vmd软件。
    然后运行`run_vmd`工具来执行编写的tcl命令。

    如果用户需要画ESP图, 你需要先调用`draw_esp`工具来绘制ESP图。
    如果用户需要画分子轨道图, 你需要先调用`draw_orbital`工具来绘制分子轨道图。

    画好图之后, 你需要调用`get_image_from_url`工具来读取图片的内容, 你需要根据图片内容调整你编写的tcl代码, 从而调整你绘制的图, 例如调整图的大小, 清晰度等。

    所有画好图之后, 请你使用Markdown格式来展示图. 首先展示分子结构图, 然后展示ESP图和分子轨道图(如果有).

    ## 作者信息
    MolPilot是由上海创智学院/华东师范大学朱通团队开发的。
    """,
    tools=[
        # orca_tool, 
        vmd_tool, 
        get_image_from_url
        ],
    )

