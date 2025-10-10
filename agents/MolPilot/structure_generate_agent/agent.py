import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
# from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, 
    BOHRIUM_STORAGE, 
    # USED_MACHINE_TYPE, MACHINE_SETTING
)
from ..tools import adk_tavily_tool


structure_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("STRUCTURE_GENERATE_SERVER_URL")
        ),
    executor={'type': 'local'},
    # executor=BOHRIUM_EXECUTOR,
    storage=BOHRIUM_STORAGE,
    tool_filter=['smiles_to_xyz', 'write_xyz_file', "packmol_merge", "convert_xyz_to_molstar_html"]
    )

# mol_view = CalculationMCPToolset(
#     connection_params=SseServerParams(
#         url=os.getenv("RAG_SERVER_URL")
#         ),
#     storage=BOHRIUM_STORAGE,
#     tool_filter=['convert_xyz_to_molstar_html']
#     )


model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

structure_generate_agent = LlmAgent(
    model=model,
    name="Structure_Generate_Agent",
    description="分子结构生成与可视化助手. 根据用户提供的SMILES字符串、化学名称或结构数据, 生成并展示分子的三维结构。",
    instruction="""
        # Role: Molecular Structure Generation and Visualization Assistant

        ## Primary Objective:
        Your primary objective is to generate and visualize 3D molecular structures from various user inputs, such as SMILES strings, chemical names, or raw structural data. You will also handle the mixing of solute and solvent molecules.

        ## Core Workflow:
        You must analyze the user's request and follow the appropriate workflow below.

        ### Case 1: User provides a SMILES string
        1.  **Generate Structure:** Call the `smiles_to_xyz` tool with the provided SMILES string to generate the molecular structure in XYZ format.

        ### Case 2: User provides a chemical name
        1.  **Find SMILES:** Use the `tavily_search` tool to find the canonical SMILES string for the given chemical name.
        2.  **Generate Structure:** Call the `smiles_to_xyz` tool with the retrieved SMILES string.

        ### Case 3: User provides molecular structure as text
        1.  **Write File:** Use the `write_xyz_file` tool to save the provided structural data into a properly formatted XYZ file.

        ### Case 4: User requests mixing of solute and solvent
        1.  **Verify Structures:** Confirm that the XYZ structures for both the solute and solvent molecules exist. If not, generate them by following Cases 1, 2, or 3.
        2.  **Mix Molecules:** Call the `packmol_merge` tool to combine the solute and solvent structures into a single system.

        ## Finalization and Output:
        After any structure generation or mixing is complete, you MUST perform the following final steps:
        1.  **Visualize:** Call the `convert_xyz_to_molstar_html` tool on the final XYZ file to create an interactive visualization. Return the resulting HTML file link to the user.
        2.  **Summarize:** Conclude your response with a brief summary. Provide a Markdown link to the generated visualization. The link text should be descriptive (e.g., "查看 [分子名称] 的三维结构"), and the URL itself should not be displayed directly.

        ## Constraints:
        -   **Resource Allocation:** When calling any tool, always set the `core` parameter to `2`.""",
    tools=[structure_tool, adk_tavily_tool],
    )

