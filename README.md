# genset-dataset-gen

基于 **LangChain + Ollama + QLoRA** 的指令微调数据集生成与训练工具链。

| 文件 | 作用 |
|---|---|
| [`genset.py`](genset.py) | 扫描文档 → 本地大模型出题 → 导出 SFT 数据集 |
| [`qlora.py`](qlora.py) | 读数据集 → 4bit QLoRA 微调 → 合并 LoRA 回底模 |

---

## 1. genset.py —— 数据集生成

扫描 `data/` 目录下的 `*.txt` 文档，按 `chunk_size=512 / overlap=50` 切块，
每个文本块从 4 个角度调用本地大模型出题，自动去重、按类别配额控制，
最终导出 `instruction / input / output` 三字段的 JSON 数据集。

**工作流程**

1. **扫描文档** — `load_docs()` 读取 data 目录，按文件名关键词归类
2. **切块** — `split_chunks()` 用 RecursiveCharacterTextSplitter 分块
3. **出题** — 每块 × 4 个角度（概念定义 / 细节事实 / 原因机理 / 影响措施）
4. **解析去重** — `parse_json()` 从模型输出中提取 JSON 数组，按问题文本去重
5. **配额控制** — 每类最少 60 条、上限 70 条，总数达标且三类齐活即提前收工

**配置**

| 常量 | 默认值 | 说明 |
|---|---|---|
| `DATA_DIR` | `/home/user/workspace/data` | 源文档目录 |
| `OUT_PATH` | `.../dataset.json` | 输出路径 |
| `MODEL` | `qwen3:8b` | 以 `ollama list` 实际输出为准 |
| `CHUNK_SIZE` / `OVERLAP` | 512 / 50 | 切块大小与重叠 |
| `TARGET` | 200 | 目标总条数 |
| `CAT_MIN` / `CAT_MAX` | 60 / 70 | 每类配额 |

**说明**

- Prompt 末尾的 `/no_think` 用于关闭 qwen3 的思考模式，避免 `think` 标签污染 JSON 输出
- `parse_json()` 对模型输出做了容错：剥离 think 标签、去 markdown 代码围栏、正则兼容截断的数组

---

## 2. qlora.py —— QLoRA 微调

读 `dataset.json`，按提示词模板拼成训练文本，用 4bit（nf4）加载底模挂 LoRA 训练，
最后把 LoRA 权重合并回 bf16 底模，产出可直接 `from_pretrained` 加载的模型。

**关键配置**

| 常量 | 默认值 | 说明 |
|---|---|---|
| `BASE_MODEL` | `/home/user/workspace/QwenPretrain` | 底模路径 |
| `DATA_PATH` | `.../dataset.json` | 训练数据 |
| `TEMPLATE` | `/home/user/workspace/tmp.doc` | 提示词模板，`{instruction}` 为占位符 |
| `LORA_R` / `LORA_ALPHA` | 8 / 16 | 秩与缩放系数 |
| `TARGET_MODULES` | `q_proj,k_proj,v_proj,o_proj` | 四个注意力层 |
| `LR` / `EPOCHS` | 2e-4 / 3 | 学习率与训练轮数 |
| `BATCH_SIZE` / `GRAD_ACCUM` | 4 / 4 | 等效 batch = 16 |
| `USE_4BIT` | `True` | nf4 量化加载 |

**几条实测踩过的坑**

- 训练文本末尾**必须补 `tok.eos_token`**，否则模型学会"永远不停"，推理时收不住
- 合并 LoRA 时**必须用 bf16 底模**，直接对 4bit 模型 merge 会得到垃圾权重
- 开了梯度检查点就**必须关 `use_cache`**
- `SFTConfig` / `SFTTrainer` 的参数名跨版本会变（`max_seq_length` vs `max_length`、
  `tokenizer` vs `processing_class`），脚本用 `inspect.signature` 做了自适应

**用法**

```bash
python qlora.py --dry-run   # 只跑 1 步，验证脚本能通
python qlora.py             # 完整训练 + 合并
```

---

## 依赖

```bash
pip install langchain-ollama langchain-text-splitters
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

数据集生成需要本地 Ollama 并已拉取模型：

```bash
ollama pull qwen3:8b
```
