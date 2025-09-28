import os

c16_list = ["c16_m16_cpu","c32_m64_cpu","c16_m64_cpu","c16_m128_cpu"]
cpu_description = ["设置核心数为8",]
USED_MACHINE_TYPE = c16_list[1]
MACHINE_SETTING = cpu_description[0]

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
            # "image_address": "registry.dp.tech/dptech/dp/native/prod-13364/orca-autode:0.0.1",
            "image_address": "registry.dp.tech/dptech/dp/native/prod-13364/autots:0.1.0",
            # "image_address": "registry.dp.tech/dptech/orca:5.0.4",
            "platform": "ali",
            "machine_type": USED_MACHINE_TYPE
            }
        },
        "resources": {
            "envs": {}
        }
    }
