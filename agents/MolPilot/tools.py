import os
import json
import base64
from typing import Optional, TypedDict, List, Dict, Any

from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults

from google.adk.tools import ToolContext, FunctionTool
from google import genai
from google.genai import types
import requests


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