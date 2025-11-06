import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .orca_analysis_agent import orca_analysis_agent
# from .pyscf_analysis_agent import pyscf_analysis_agent

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

report_agent = LlmAgent(
    model=model, 
    name="Report_Agent",
    description="报告生成调度者, 根据不同的计算化学软件, 调用不同的分析子代理.",
    instruction="""
        # Role: Report Generation Coordinator

        ## Primary Objective:
        Your primary objective is to coordinate the generation of comprehensive reports based on the outputs from various quantum chemistry software packages.
        You will delegate the analysis tasks to specialized sub-agents depending on the software used in the calculations.

        ## 作者信息
        MolPilot是由上海创智学院/华东师范大学朱通团队开发的。""",
        
    sub_agents=[
        orca_analysis_agent,
        # pyscf_analysis_agent,
    ]
    )

