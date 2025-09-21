## 环境安装

```bash
pip install -r requirements.txt
```

## 运行 Google-adk

运行前需要配置好.env中的参数. 

如果需要把任务提交到玻尔上计算，需要设置`constant.py`中的`BOHRIUM_ACCESS_KEY`和`BOHRIUM_PROJECT_ID`.

```bash

cd path_to/build-your-agent/agents
adk web
```