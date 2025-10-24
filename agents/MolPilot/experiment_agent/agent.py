import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .sub_agents import orca_agent
from .sub_agents import pyscf_agent


model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

experiment_agent = LlmAgent(
    model=model, 
    name="quantum_chemistry_experiment_agent",
    description="量子化学计算实验助手. 负责解析用户意图, 然后将任务交给子代理.",
    instruction=f"""
    # Role: Quantum Chemistry Experiment Assistant

    ## Primary Objective:
    Your primary objective is to understand the user's computational chemistry goals, then deliver the task to sub-agent.
    
    当用户需要结构优化, 计算过渡态,光谱时, 使用orca_agent处理任务.
    当用户需要进行量子化学计算, 例如计算分子轨道, 电子性质时, 使用pyscf_agent处理任务.
    """,
    sub_agents=[
            orca_agent,
            pyscf_agent,
            # ase_agent,
        ],
    )