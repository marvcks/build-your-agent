import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams

from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, 
    BOHRIUM_STORAGE,
    )


rest_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        # url="http://0.0.0.0:50002/sse"
        url="http://39.98.161.104:50001/sse"
        ),
    # executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=['run_rest', 'read_rest_log', 'run_rest_interaction']
    )


model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

rest_agent = LlmAgent(
    model=model, 
    name="Rest_Agent",
    description="REST量子化学计算任务管理器.负责解析用户意图,准备计算输入,并使用REST执行结构优化,频率分析,电子性质计算等任务.",
    instruction=f"""
    ## 任务
    1. 使用`run_rest`工具运行REST计算任务
    2. 使用`parse_rest_log`解析REST计算任务日志.
    3. 将解析后的结果转交给`Report_Agent`下的`Rest_Analysis_Agent`进行处理.

    ## specific workflow
    ### nonbonded interaction energy scan
        1. 使用`run_rest_interaction`计算能量扫描.只允许调用一次.
        2. 将任务转交给`Report_Agent`下的`Rest_Analysis_Agent`进行处理.
    """,
    tools=[rest_tool],
    )


