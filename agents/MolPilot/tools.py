import os
import json
import base64
from typing import Optional, TypedDict, List, Dict, Any
from pathlib import Path
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults

from google.adk.tools import ToolContext, FunctionTool
from google import genai
from google.genai import types
import requests
from dp.agent.server.storage.http_storage import HTTPSStorage
import uuid


# Instantiate the LangChain tool
tavily_tool_instance = TavilySearchResults(
    max_results=2,
    # search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

# Wrap it with LangchainTool for ADK
adk_tavily_tool = LangchainTool(tool=tavily_tool_instance)


async def get_image_from_url(image_url: str, fig_name: str, tool_context: ToolContext):
    """
    Get image from url.

    Args:
        image_url (str): The url of the image.
        fig_name (str): The name of the figure. For example, "esp" for electrostatic potential.

    Returns:
        dict: The result.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content
        
        artifact_name = f"generated_image_" + fig_name + ".png"

        # Save as ADK artifact (optional, if still needed by other ADK components)
        report_artifact = types.Part.from_bytes(
            data=image_bytes, mime_type="image/png"
        )

        await tool_context.save_artifact(artifact_name, report_artifact)
        print(f"Image also saved as ADK artifact: {artifact_name}")

        return {
            "status": "success",
            "message": f"Image generated. ADK artifact: {artifact_name}.",
            "artifact_name": artifact_name,
        }
    except Exception as e:
        return {"status": "error", "message": f"No images generated.  {e}"}


async def get_image(fig_name: str, tool_context: ToolContext):
    """
    Get image artifact from ADK.

    Args:
        fig_name (str): The name of the figure. For example, "esp" for electrostatic potential.

    Returns:
        dict: The result.
    """
    try:
        
        artifact_name = (
            f"generated_image_" + fig_name + ".png"
        )
        artifact = await tool_context.load_artifact(artifact_name)
    
        return {
            "status": "success",
            "message": f"Image artifact {artifact_name} successfully loaded."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading artifact {artifact_name}: {str(e)}"
        }
    


def upload_file_to_bohr(file_path: str) -> Dict[str, Any]:
    """
    将本地文件上传到 Bohr 对象存储服务，并获取可公开访问的 URL。
    """
    
    # 1. 检查依赖
    if HTTPSStorage is None:
        return {
            "status": "error", 
            "error_message": "无法导入 HTTPSStorage，请检查项目依赖 (dp.agent.server.storage)。"
        }

    # 2. 检查本地文件是否存在
    local_path = Path(file_path)
    if not local_path.exists() or not local_path.is_file():
        return {
            "status": "error", 
            "error_message": f"文件不存在或路径无效: {file_path}"
        }

    # 3. 获取配置 (直接读取独立变量)
    access_key = os.getenv("BOHRIUM_ACCESS_KEY")
    project_id = os.getenv("BOHRIUM_PROJECT_ID")
    app_key = os.getenv("BOHRIUM_APP_KEY", "agent") # 默认为 agent
    
    # 兼容旧逻辑：如果没有具体配置，尝试只读取类型
    plugin_type_str = os.getenv("HTTP_PLUGIN_TYPE")

    storage_config = {}

    # 4. 组装配置字典
    if access_key and project_id:
        try:
            # 自动组装字典，project_id 需要转为整数
            storage_config = {
                "type": "bohrium",
                "access_key": access_key,
                "project_id": int(project_id),
                "app_key": app_key
            }
        except ValueError:
             return {
                "status": "error", 
                "error_message": "BOHRIUM_PROJECT_ID 必须是数字。"
            }
    elif plugin_type_str:
        # 降级方案
        storage_config = {"type": plugin_type_str}
    else:
        return {
            "status": "error", 
            "error_message": "未找到存储配置。请在 .env 中设置 BOHRIUM_ACCESS_KEY 和 BOHRIUM_PROJECT_ID。"
        }

    try:
        # 5. 初始化存储客户端
        # print(f"DEBUG: 使用配置初始化存储: {storage_config}") # 调试用
        storage = HTTPSStorage(plugin=storage_config)

        # 6. 执行上传
        file_key = f"uploads/{uuid.uuid4()}/{local_path.name}"
        upload_result = storage._upload(file_key, str(local_path))

        # 7. 处理结果
        if upload_result:
            if not str(upload_result).startswith(('http://', 'https://')):
                upload_url = f"https://{upload_result}"
            else:
                upload_url = upload_result
            
            return {
                "status": "success",
                "url": upload_url,
                "filename": local_path.name
            }
        else:
            return {
                "status": "error", 
                "error_message": "存储服务返回结果为空，上传可能失败。"
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "error_message": f"上传过程中发生意外错误: {str(e)}"
        }
    

if __name__ == "__main__":
    # 测试上传功能
    from dotenv import load_dotenv
    load_dotenv()
    test_file = "test_upload.txt"
    with open(test_file, "w") as f:
        f.write("这是一个测试文件，用于验证上传功能。")

    result = upload_file_to_bohr(test_file)
    print(result)