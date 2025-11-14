import os

image_address = "registry.dp.tech/dptech/dp/native/prod-13364/molpilot-server:0.4.0"

BOHRIUM_STORAGE = {
    "type": "https",
    "plugin": {
        "type": "bohrium",
        "access_key": os.getenv("BOHRIUM_ACCESS_KEY"),
        "project_id": os.getenv("BOHRIUM_PROJECT_ID"),
        "app_key": "agent"
        }
    }
    
BOHRIUM_EXECUTOR = {
  "type": "dispatcher",
  "machine": {
    "batch_type": "OpenAPI",
    "context_type": "OpenAPI",
    "remote_profile": {
      "access_key": os.getenv("BOHRIUM_ACCESS_KEY"),
      "project_id": os.getenv("BOHRIUM_PROJECT_ID"),
      "app_key": "agent",
      "image_address": image_address,
      "platform": "ali",
      "machine_type": "c32_m128_cpu"
    }
  },
  "resources": {
    "envs": {
      "BOHRIUM_PROJECT_ID": os.getenv("BOHRIUM_PROJECT_ID")
    }
  }
}