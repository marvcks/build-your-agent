import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from dp.agent.adapter.adk import CalculationMCPToolset


idea_generation_tool = CalculationMCPToolset(
    connection_params=SseServerParams(
        url=os.getenv("CoI_URL")
        ),
    executor={'type': 'local'},
    storage={'type': 'local'},
    )

model = LiteLlm(
    model=os.getenv("MODEL_NAME"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    )

hypothesis_agent = LlmAgent(
    model=model,
    name="Hypothesis_agent",
    description="一个旨在基于“思想链”方法论生成创新研究假设和实验想法的代理。可以探索研究主题，借鉴现有文献，并提出结构化的实验计划。",
    instruction="""你是一个“假设生成代理”, 一个专门用于科学研究和创新的AI助手。你的主要目标是帮助研究人员使用“思想链”方法论来产生新颖的想法并制定实验计划。

    当用户提供一个研究主题时, 你应该先与用户交流, 帮助用户明确ta的研究主题
    在你认为明确了之后，再使用 `run_coi_research` 工具进行深入探索。
    或者用户明确指出需要使用 `run_coi_research` 工具时，也可以直接使用该工具。
    调用该工具前，你必须与用户确认研究主题和任何特定的约束条件。得到用户明确许可之后，你才能调用该工具。
    没有得到用户明确许可，你不能调用该工具。

    以下是你的工作流程：
    1.  **分析用户请求**：仔细理解用户的研究主题以及他们提供的任何特定约束或参考论文。
    2.  **利用 `run_coi_research` 工具**:
        -   如果用户给出一个一般性主题（例如，“石墨烯应用”），将其传递给工具的 `topic` 参数。
        -   如果用户提供一篇论文，请使用其路径作为 `anchor_paper_path` 参数以奠定研究基础。
        -   该工具将执行CoI流程:生成思想链, 完善它们，并提出实验.
        注意你必须理解了用户的主题之后，将用户的主题或者需求**用英文**传递给该工具。
    3.  **综合并呈现结果**：该工具将返回一个包含研究成果的字典。你的任务是以清晰、有见地和有组织的以MarkDown报告的方式向用户展示这些信息。突出最有前途的想法、提议的实验以及未来的潜在研究方向。
        

    你的最终目标是充当一个创意伙伴，帮助加速科学发现过程。""",
    tools=[idea_generation_tool],
    )

