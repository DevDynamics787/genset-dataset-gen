# genset-dataset-gen

基于 **LangChain + Ollama** 的指令微调数据集（SFT）自动生成工具。

## 功能

扫描 `data/` 目录下的 `*.txt` 文档，按 [chunk_size=512 / overlap=50] 切块，
每个文本块从 4 个角度调用本地大模型出题，自动去重、按类别配额控制，
最终导出 `instruction / input / output` 三字段的 JSON 数据集。

## 工作流程

1. **扫描文档** — `load_docs()` 读取 data 目录，按文件名关键词归类
2. **切块** — `split_chunks()` 用 RecursiveCharacterTextSplitter 分块
3. **出题** — 每块 × 4 个角度（概念定义 / 细节事实 / 原因机理 / 影响措施）
4. **解析去重** — `parse_json()` 从模型输出中提取 JSON 数组，按问题文本去重
5. **配额控制** — 每类最少 60 条、上限 70 条，总数达标且三类齐活即提前收工

## 依赖

```bash
pip install langchain-ollama langchain-text-splitters
```

需要本地运行 Ollama 并已拉取模型：

```bash
ollama pull qwen3:8b
```

## 配置

脚本顶部常量按需修改：

| 常量 | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `/home/user/workspace/data` | 源文档目录 |
| `OUT_PATH` | `.../dataset.json` | 输出路径 |
| `MODEL` | `qwen3:8b` | 以 `ollama list` 实际输出为准 |
| `CHUNK_SIZE` / `OVERLAP` | 512 / 50 | 切块大小与重叠 |
| `TARGET` | 200 | 目标总条数 |
| `CAT_MIN` / `CAT_MAX` | 60 / 70 | 每类配额 |

## 运行

```bash
python genset.py
```

## 说明

- Prompt 末尾的 `/no_think` 用于关闭 qwen3 的思考模式，避免 ` thinking` 标签污染 JSON 输出
- `parse_json()` 对模型输出做了容错：剥离 think 标签、去 markdown 代码围栏、正则兼容截断的数组
