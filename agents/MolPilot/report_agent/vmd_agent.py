import os
from langchain_core.tools import tool
import requests

from dp.agent.adapter.adk import CalculationMCPToolset

from google.adk.agents import LlmAgent, Agent, LoopAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.tools import ToolContext, FunctionTool
from google import genai
from google.genai import types


async def get_image_from_url(image_url: str, tool_context: ToolContext):
    """
    Get image from url.

    Args:
        image_url (str): The url of the image.

    Returns:
        dict: The result.
    """
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


def set_score(tool_context: ToolContext, score: str) -> str:
    """
    Set the score of the visualization evaluation.

    Args:
        tool_context (ToolContext): The tool context.
        score (str): The score. Must be one of "pass" or "fail".

    Returns:
        str: The score.
    """
    print(f"score is {score}")
    tool_context.state["score"] = score


def check_condition_and_escalate_tool(tool_context: ToolContext) -> dict:
    """Checks the loop termination condition and escalates if met or max count reached."""
 

    # Increment loop iteration count using state
    current_loop_count = tool_context.state.get("loop_iteration", 0)
    current_loop_count += 1
    tool_context.state["loop_iteration"] = current_loop_count

    # Define maximum iterations
    max_iterations = 3

    # Get the condition result set by the sequential agent from state
    total_score = tool_context.state.get("score", "fail")

    condition_met = total_score == "pass"

    response_message = f"Check iteration {current_loop_count}: Sequential condition met = {condition_met}. "

    # Check if the condition is met OR maximum iterations are reached
    if condition_met:
        print("  Condition met. Setting escalate=True to stop the LoopAgent.")
        tool_context.actions.escalate = True
        response_message += "Condition met, stopping loop."
    elif current_loop_count >= max_iterations:
        print(
            f"  Max iterations ({max_iterations}) reached. Setting escalate=True to stop the LoopAgent."
        )
        tool_context.actions.escalate = True
        response_message += "Max iterations reached, stopping loop."
    else:
        print(
            "  Condition not met and max iterations not reached. Loop will continue."
        )
        response_message += "Loop continues."

    return {"status": "Evaluated scoring condition", "message": response_message}

check_tool_condition = FunctionTool(func=check_condition_and_escalate_tool)


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

vmd_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=[
        'get_vmd_manual', 'run_vmd', 'draw_esp', 'draw_orbital', 'draw_colorbar_for_esp'
        ]
    )


draw_agent = LlmAgent(
    model=model, 
    name="Draw_Agent",
    description="分子可视化助手, 主要是用VMD软件来可视化分子静电势,分子轨道,Fukui函数.",
    instruction=f"""
    你是一个分子可视化助手, 主要是用VMD软件来可视化分子静电势,分子轨道,Fukui函数。

    1.用`get_vmd_manual`工具来获取编写tcl命令的参考资料, 编写tcl命令以调用vmd软件。
    2.如果用户需要画ESP图, 调用`draw_esp`工具来绘制ESP图.
      如果用户需要画HOMO,LUMO轨道图, 调用`draw_orbital`工具来绘制分子轨道图.
      如果用户需要画Fukui函数图, 多次调用`draw_orbital`工具来绘制Fukui函数图,f-,f+,f0,fdd.
      如果用户需要画分子结构图, 调用`run_vmd`工具来绘制分子结构图.
    3. 调用`get_image_from_url`工具加载图片.

    如果你收到了对之前做出来的图片的评估, 你需要根据评估内容,调整你编写的tcl代码, 从而调整你绘制的图.
    """,
    tools=[vmd_tool, get_image_from_url],
    output_key="output_image"
    )


evaulator_agent = LlmAgent(
    model=model, 
    name="Evaulator_Agent",
    description="评估分子可视化结果的质量并给出改进建议",
    instruction=f"""
    你是一个分子可视化结果评估助手, 主要是评估分子可视化结果的质量并给出改进建议。

    1. 调用'get_image'工具读取图片的内容, 你需要根据图片内容评估可视化结果的质量。
    2. 调用`set_score`工具来设置给图片打分，必须为"pass"或"fail"。

    评估可视化结果的质量时, 你需要考虑以下几个方面:
    - 图片中的分子大小(scale by x.x)是否合适
    - 图片中展示的分子或者等值面的角度是否合适, 等值面是否完整的展示出来了
    - 等值面大小(set isoval x.xx)是否合适

    如果你给出了"fail", 请你给出具体的改进建议.不需要提问.

    """,
    tools=[set_score, get_image],
    output_key="visualization_evaluation")


checker_agent_instance = Agent(
    name="checker_agent",
    model=model,
    instruction="""
    判断可视化结果的质量是否符合要求.

    Use the 'check_condition_and_escalate_tool' to evaluate if the score is "pass" or if loop has execeed the MAX_ITERATIONS.

    If the score is "pass" or if loop has execeed the MAX_ITERATIONS,
    the loop will be terminated.

    If the score is not "pass" or if loop has not execeed the MAX_ITERATIONS,
    the loop will continue.
    """,
    tools=[check_tool_condition],
    output_key="checker_output"
    )


image_generation_scoring_agent = SequentialAgent(
    name="image_generation_scoring_agent",
    description=(
        """
        生成图片, 评估图片质量, 并给出改进建议。
        1. 调用Draw_Agent来生成图片
        2. 调用Evaulator_Agent来评估图片质量
        """
    ),
    sub_agents=[
        draw_agent, 
        evaulator_agent
        ],
    )


vmd_agent = LoopAgent(
    name="VMD_Agent",
    description="分子可视化助手, 主要是用VMD软件来可视化分子静电势,分子轨道,Fukui函数.",
    sub_agents=[
        image_generation_scoring_agent, 
        checker_agent_instance
        ],
    )