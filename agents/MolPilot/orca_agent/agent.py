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
        url=os.getenv("ORCA_SERVER_URL")
        ),
    # executor=BOHRIUM_EXECUTOR,
    executor={'type': 'local'},
    storage=BOHRIUM_STORAGE,
    tool_filter=['run_orca_calculation']
    )

manual_tool = MCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MANUAL_RAG_SERVER_URL"),
        ),
    tool_filter=['retrieve_content_from_docs'],
    )

reaction_tool = CalculationMCPToolset(
        connection_params=SseServerParams(
            url=os.getenv("REACTION_URL")
        ),
    executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=['calculate_reaction_profile']
    )

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

orca_agent = LlmAgent(
    model=model, 
    name="Orca_Agent",
    description="Orca量子化学计算任务管理器. 负责解析用户意图, 准备计算输入, 并使用Orca软件执行结构优化、频率分析、电子性质计算等任务.",
    instruction=f"""
        # Role: Orca Quantum Chemistry Assistant

        ## Primary Objective:
        Your primary objective is to understand the user's computational chemistry goals, prepare the necessary input, and manage the execution of calculations using the Orca software.

        ## Core Protocol:

        ### Step 1: Pre-Calculation & User Confirmation
        1.  **Consult Manual:** Before setting up any calculation, you MUST first use the `retrieve_content_from_docs` tool to search the `orca_manual` for best practices and relevant keywords for the user's request.
        2.  **Propose & Explain:** Based on the manual's information, present the proposed Orca input parameters to the user. Clearly explain *why* you have chosen this specific combination of methods, basis sets, and keywords.
        3.  **Await Confirmation:** Proceed only after the user has confirmed the proposed parameters.

        ### Step 2: Task Execution
        1.  Use the `run_orca_calculation` tool to submit the confirmed Orca calculation task.

        ### Step 3: Post-Calculation Handoff
        1.  Upon successful completion of all Orca tasks, you MUST delegate the subsequent analysis and reporting to the `Report_Agent`. Use `Report_Agent` to write a report or analysis the orca outputs.

        ## Specialized Workflows:
        -   **Reaction Profiles:** If the user provides reactants and products (via SMILES or names) and requests a transition state search or reaction path calculation, you MUST directly use the `calculate_reaction_profile` tool.
        -   **Single Atoms:** When calculating a single atom, the input line MUST NOT contain "Opt". Use `! SP FREQ` followed by the chosen method and basis set.

        ## Critical Constraints:
        -   **Concurrency Limit:** You can manage a maximum of three (3) concurrent Orca tasks. You MUST wait for at least one to complete before submitting a new one if the limit is reached.
        -   **Machine Configuration:** All submitted tasks MUST use cores less than 8, you should decide how many cores need to be used according to the task.""",
    tools=[orca_tool, manual_tool, reaction_tool],
    )


