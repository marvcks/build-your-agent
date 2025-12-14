import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .sub_agents import orca_agent
from .sub_agents import pyscf_agent
from .sub_agents import rest_agent
from .sub_agents import multiwfn_agent
from .sub_agents import reaction_agent



model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

experiment_agent = LlmAgent(
    model=model, 
    name="Experiment_Agent",
    description="量子化学计算实验助手. 能够调用ORCA, PySCF, REST, Multiwfn软件进行量子化学计算.其中PySCF中包含了深度学习泛函Skala.Multiwfn能够对波函数文件进行后处理.能够进行化学反应枚举和计算.",
    instruction=f"""
    # Role: Quantum Chemistry Experiment Assistant

    ## Primary Objective:
    Your primary objective is to understand the user's computational chemistry goals, then deliver the task to sub-agent.
    
    使用"Orca_Agent"进行结构优化, 计算过渡态,光谱.
    使用"PySCF_Agent"进行量子化学计算, 例如计算分子轨道, 电子性质.
    使用"Rest_Agent"进行REST计算.
    使用"Multiwfn_Agent"处理波函数文件,例如计算分子的静电势,HOMO,LUMO轨道,Fukui函数.
    使用"Reaction_Agent"进行化学反应枚举和计算.
    """,
    sub_agents=[
            orca_agent,
            pyscf_agent,
            rest_agent,
            multiwfn_agent,
            reaction_agent,
            # ase_agent,
        ],
    )