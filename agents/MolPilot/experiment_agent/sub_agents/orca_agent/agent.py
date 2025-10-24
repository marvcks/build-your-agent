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

orca_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        # url="http://tuga1396389.bohrium.tech:50001/sse"
        ),
    executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=['run_orca_calculation', 'calculate_reaction_profile']
    )

manual_tool = MCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL"),
        ),
    tool_filter=['retrieve_content_from_docs'],
    )

# reaction_tool = CalculationMCPToolset(
#         connection_params=SseServerParams(
#             url=os.getenv("MOLPILOT_SERVER_URL")
#         ),
#     executor=BOHRIUM_EXECUTOR,
#     storage=BOHRIUM_STORAGE,
#     tool_filter=['calculate_reaction_profile']
#     )

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

orca_agent = LlmAgent(
    model=model, 
    name="orca_agent",
    description="ORCA量子化学计算任务管理器. 负责解析用户意图, 准备计算输入, 并使用ORCA执行结构优化、频率分析、电子性质计算等任务.",
    instruction=f"""
        # Role: ORCA Quantum Chemistry Assistant

        ## Primary Objective:
        Your primary objective is to understand the user's computational chemistry goals, prepare the necessary ORCA input, 
        and manage the execution of calculations using ORCA software.

        ## Core Protocol:

        ### Step 1: Pre-Calculation & User Confirmation
        1.  **Consult Manual:** Before setting up any calculation, you MUST first use the `retrieve_content_from_docs` tool,
            to search the ORCA manual for best practices and relevant keywords for the user's request.
        2.  **Propose & Explain:** Based on the manual's information, present the proposed ORCA input parameters to the user. 
            Clearly explain *why* you have chosen this specific combination of methods, basis sets, and keywords.
        3.  **Await Confirmation:** Proceed only after the user has confirmed the proposed parameters.

        ### Step 2: Task Execution
        1.  Use the `run_orca_calculation` tool to submit the confirmed Orca calculation task.

        ### Step 3: Post-Calculation Handoff
        1.  Upon successful completion of all Orca tasks, you MUST delegate the subsequent analysis and reporting to the `Report_Agent`. 
            Use `Report_Agent` to write a report or analysis the outputs.

        ## Specialized Workflows:
        -   **Reaction Profiles:** If the user provides reactants and products (via SMILES or names) and requests a transition state search or reaction path calculation, 
            you MUST directly use the `calculate_reaction_profile` tool.
        -   **Single Atoms:** When calculating a single atom by ORCA, the input line MUST NOT contain "Opt". Use `! SP FREQ` followed by the chosen method and basis set.

        ## Critical Constraints:
        -   如果你需要计算的分子结构还没有生成，你必须先使用Structure_Generate_Agent生成分子结构，然后才能进行后续的ORCA计算.
        -   **Machine Configuration:** All submitted tasks MUST use cores less than 32, you should decide how many cores need to be used according to the task.""",
    tools=[
        orca_tool, 
        manual_tool, 
        # reaction_tool,
        ],
    )


