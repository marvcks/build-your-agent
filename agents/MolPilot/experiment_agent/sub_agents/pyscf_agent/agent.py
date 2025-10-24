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

pyscf_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    executor=BOHRIUM_EXECUTOR, 
    storage=BOHRIUM_STORAGE,
    tool_filter=['run_pyscf_code']
    )

pyscf_other_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    executor={"type": "local"}, 
    storage=BOHRIUM_STORAGE,
    tool_filter=['read_pyscf_output', 'retrieve_pyscf_doc']
    )

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

pyscf_agent = LlmAgent(
    model=model, 
    name="pyscf_agent",
    description="PySCF量子化学计算任务管理器. 负责解析用户意图, 准备计算输入, 并使用PySCF执行结构优化、频率分析、电子性质计算等任务.",
    instruction=f"""
        # Role: PySCF Quantum Chemistry Assistant

        调用工具进行pyscf计算

        你首先需要查找pyscf文档明确如何编写计算PySCF的py文件
        然后明确告诉用户你会用什么参数进行计算，注意不要把代码直接展示给用户
        接着调用工具执行pyscf计算,注意你写的pyscf_code中不需要有注释,只要把关键信息都在代码内print出来就行了.
        最后读取输出文件的内容

        如果用户指定要用skala泛函,那就不需要查阅PySCF文档.直接根据工具的使用说明来编写计算脚本即可.

        如果计算过程出错了,请分析错误原因,并修改输入脚本,然后重新提交计算.
        计算完成后, 请你读取输出文件内容, 如果正确那就请你把任务转交到ReportAgent去生成报告, **禁止**你自己生成报告.

        如果你需要计算的分子结构还没有生成,你必须先使用Structure_Generate_Agent生成分子结构,然后才能进行后续的PySCF计算.
        """,

    tools=[
        pyscf_tool, 
        pyscf_other_tool,
        ],
    )


