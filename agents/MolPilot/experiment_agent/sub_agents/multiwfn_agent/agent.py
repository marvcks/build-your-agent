import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, 
    BOHRIUM_STORAGE,
    )


multiwfn_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=[
        'run_multiwfn_esp', 'run_multiwfn_orbital', 'run_multiwfn_fukui'
        ]
    )

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

multiwfn_agent = LlmAgent(
    model=model, 
    name="Multiwfn_Agent",
    description="Multiwfn波函数文件分析器.使用Multiwfn执行波函数文件分析任务,包括计算分子的静电势,HOMO,LUMO轨道,Fukui函数.",
    instruction=f"""
    你是一个Multiwfn波函数文件分析器.负责解析用户意图,准备计算输入,并使用Multiwfn执行波函数文件分析任务.

    ## 任务

    ### 计算分子轨道
    
    当需要计算分子的轨道时, 使用`run_multiwfn_orbital`工具.
    当需要计算分子的Fukui函数时, 使用`run_multiwfn_fukui`工具.
    当需要计算分子的静电势时, 使用`run_multiwfn_esp`工具.

    你的任务完成之后,你**必须**调度"Report_Agent"进行报告总结.
    要求"Report_Agent"读取生成的图片,并根据图片内容进行分析,总结计算结果.

    你不能自己给出报告.
    """,
    tools=[multiwfn_tool],
    )


