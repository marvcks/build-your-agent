import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from .orca_analysis_agent import orca_analysis_agent
from .rest_analysis_agent import rest_analysis_agent
# from .vmd_agent import vmd_agent
from .multiwfn_analysis_agent import multiwfn_analysis_agent

# from .pyscf_analysis_agent import pyscf_analysis_agent

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

report_agent = LlmAgent(
    model=model, 
    name="Report_Agent",
    description="报告生成者,负责处理总结ORCA,PySCF,REST,Multiwfn的计算结果.",
    instruction="""
        # Role: Report Generation Coordinator

        ## Primary Objective:
        Your primary objective is to coordinate the generation of comprehensive reports based on the outputs from various quantum chemistry software packages. You cannot generate reports on your own.
        You will delegate the analysis tasks to specialized sub-agents depending on the software used in the calculations.

        调用"Orca_Analysis_Agent"处理ORCA的计算结果
        调用"Rest_Analysis_Agent"处理REST的计算结果
        调用"Multiwfn_Analysis_Agent"处理Multiwfn的计算结果


        ## 作者信息
        MolPilot是由上海创智学院/华东师范大学朱通团队开发的。""",
        
    sub_agents=[
        orca_analysis_agent,
        # vmd_agent,
        rest_analysis_agent,
        # pyscf_analysis_agent,
        multiwfn_analysis_agent,
    ]
    )

