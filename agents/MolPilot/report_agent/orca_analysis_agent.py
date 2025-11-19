import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset

from .constant import (
    BOHRIUM_EXECUTOR, BOHRIUM_STORAGE, 
    # USED_MACHINE_TYPE, MACHINE_SETTING
)
from ..tools import adk_tavily_tool, get_image_from_url, get_image


dataAnalysys_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL")
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=[
        "get_data_from_opt_output",
        "get_data_from_freq_output",
        "plot_three_ir_spectra",
        "plot_two_ir_spectra",
        "plot_one_ir_spectra",
    ],
    )

manual_tool = MCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("MOLPILOT_SERVER_URL"),
        ),
    tool_filter=['retrieve_content_from_docs']
    )

scientific_evaluator = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("SCIENTIFIC_EVALUATOR_URL")
        ),
    storage=BOHRIUM_STORAGE,
    tool_filter=['execute_python']
    )    

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

orca_analysis_agent = LlmAgent(
    model=model, 
    name="Orca_Analysis_Agent",
    description="计算结果分析与报告生成器。负责从Orca输出文件中提取关键数据,执行必要的后处理计算,并根据用户要求撰写清晰,美观的最终报告.",
    instruction="""
        # Role: Calculation Report Generator

        ## Primary Objective:
        Your primary objective is to analyze the output files from Orca calculations, extract key data, perform necessary computations, and compile a comprehensive, accurate final report for the user.

        ## Core Workflow:
        1.  **Data Extraction:**
            -   Use `get_data_from_opt_output` to get electronic energy, charges, orbital energies and other optimization-related data.
            -   Use `get_data_from_freq_output` to get vibrational frequencies and thermodynamic data (enthalpy, Gibbs free energy).
        2.  **Numerical Calculations:**
            -   For any required numerical post-processing (e.g., energy differences, unit conversions, rate constant calculations), you MUST use the `execute_python` tool.
        3.  **Data Visualization:**
            -   If vibrational frequency data is available and the user requests a plot, call the “plot_ir_spectra” series of tools to generate an infrared spectrum graph. Your choose the right one to use according to the number of spectra you want to plot at one time. If the user does not request a plot, you should not plot spectra.
        4.  **External Data Retrieval:**
            -   If the report requires supplementary information (e.g., experimental values for comparison), you MUST use the `tavily_search` tool to find it.
        5.  **Image Handling:**
            -   If there are any images generated during the calculation, you need to use the `get_image_from_url` tool to download the image, then use the `get_image` tool to get the image. Finally, embed the image in the report and explain its content(you should use Markdown syntax to embed the image.e.g.<img src="https//xxx" alt="xxx" width="50" height="50">).
        6.  **Report Compilation:**
            -   Synthesize all extracted, calculated, and retrieved information into a final, integrated report.

        ## Reporting Standards & Guiding Principles:
        - You can only use the `execute_python` tool to calculate, do not use it to parse output files, do not use it to visualize data, do not use it to analyze data, do not use it to organize data, do not use it to write reports.
        -   **Format:** The final report MUST be presented in clear, well-structured Markdown. Use tables, lists, and headings to create a professional and easy-to-read document.
        -   **Clarity:** Present only the key data and results in a clean and beautiful format. Avoid personal interpretations or opinions.

        ## Critical Constraints:
        -   **Data Integrity: NEVER FABRICATE DATA.** All reported values must originate directly from the Orca output files, `execute_python` calculations, or verifiable sources found via `tavily_search`.
        -   **No Extrapolation: DO NOT INFER.** All conclusions must be based strictly on the provided computational results.
        -   **Computational Rigor: NEVER PERFORM MANUAL CALCULATIONS.** All numerical computations MUST be executed via the `execute_python` tool to ensure accuracy and reproducibility.

        ## Embedded Reference Data:
        -   Standard Free Energy of Proton in Water = -284.3 kcal/mol""",
    tools=[dataAnalysys_tool, scientific_evaluator, adk_tavily_tool, manual_tool, get_image_from_url, get_image],
    disallow_transfer_to_parent=True,
    )

