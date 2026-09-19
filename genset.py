import os, re, json, glob, time
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
DATA_DIR = "/home/user/workspace/data"
OUT_PATH = "/home/user/workspace/model_b/data/dataset.json"  # 输出：数据集
MODEL = "qwen3:8b"        # 以 ollama list 实际输出的名字为准！
CHUNK_SIZE = 512          # 每块字数（样题要求 ≈512）
OVERLAP = 50              # 相邻块重叠字数
TARGET = 200              # 目标总条数
CAT_MIN = 60              # 每个类别的最低条数
CAT_MAX = 70              # 每个类别的上限（防止某一类刷爆、浪费时间）
PER_CALL = 3              # 每次让模型生成几条

# 文件名关键词 → 主题类别
CATEGORIES = {"次贷": "金融", "危机": "金融",
              "心包炎": "医疗", "尿毒": "医疗",
              "人工智能": "法规", "暂行办法": "法规"}


ANGLES = ["请提出一个「某个概念/术语是什么」的问题并作答",
          "请针对文中的具体数据、时间或事实提一个细节问题并作答",
          "请针对文中事件的原因、机理或背景提一个问题并作答",
          "请针对文中的影响、意义或应对措施提一个问题并作答"]

# Prompt 模板
PROMPT_TPL = """你是一个擅长总结和出题的AI助手。
请根据下面的文档片段，生成 {n} 个具体的问题及其标准答案。
要求：
1. 问题必须能在原文中找到答案，不要编造；
2. 严格按 JSON 数组格式输出，不要输出任何解释文字；
3. JSON 格式示例：
[{{"instruction": "问题", "input": "", "output": "标准答案"}}]
4. 本次出题角度：{angle}

文档片段：
{chunk}
/no_think"""


def load_docs():
    """扫描 data 目录下所有 txt，返回 [(类别, 文件名, 全文), ...]"""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        name = os.path.basename(path)
        category = "其他"
        for kw, cat in CATEGORIES.items():
            if kw in name:
                category = cat
                break
        text = open(path, encoding="utf-8").read()
        docs.append((category, name, text))
    return docs


def split_chunks(text):
    """按 chunk_size=512 / overlap=50 把长文切成小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
    )
    return splitter.split_text(text)


def parse_json(txt):
    """把模型输出里的 JSON 数组抠出来 → list；解析失败返回 []"""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    txt = txt.replace("```json", " ").replace("```", " ")
    cand = re.findall(r"\[.*?\]", txt, flags=re.S)
    m = re.search(r"\[.*\]", txt, flags=re.S)
    if m and m.group(0) not in cand:
        cand.append(m.group(0))
    items = []
    for b in cand:
        try:
            data = json.loads(b)
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for d in data:
            if not isinstance(d, dict):
                continue
            q = str(d.get("instruction", "") or "").strip()
            a = str(d.get("output", "") or "").strip()
            if q and a:
                items.append({"instruction": q, "input": str(d.get("input", "") or ""), "output": a})
    return items


def main():
    t0 = time.time()
    docs = load_docs()
    print("=== 扫描到文档 ===")
    for cat, name, text in docs:
        print(f"  [{cat}] {name}   长度 {len(text)} 字")

    # 产能引擎：每篇文档切块，每块 × 4 个角度 = 4 次调用
    jobs = []
    for cat, name, text in docs:
        chunks = split_chunks(text)
        print(f"  {name} → 切成 {len(chunks)} 块")
        for ci, chunk in enumerate(chunks):
            for ai, angle in enumerate(ANGLES):
                jobs.append((cat, name, ci, ai, angle, chunk))
    print(f"=== 理论产能：{len(jobs)} 次调用 × {PER_CALL} 条 = 最多 {len(jobs) * PER_CALL} 条 ===")

    llm = ChatOllama(model=MODEL, temperature=0.7, num_predict=1024)

    dataset = []
    seen = set()          # 去重用：记录已出现的问题
    per_cat = {}          # 每类已生成条数
    for i, (cat, name, ci, ai, angle, chunk) in enumerate(jobs, 1):
        if len(dataset) >= TARGET and all(per_cat.get(c, 0) >= CAT_MIN for c in ("金融", "医疗", "法规")):
            print(">>> 总数达标且三类都 ≥60，提前收工")
            break
        if per_cat.get(cat, 0) >= CAT_MAX:
            continue       # 这一类够了，跳过，把时间留给没够的类
        prompt = PROMPT_TPL.format(n=PER_CALL, angle=angle, chunk=chunk)
        try:
            out = llm.invoke(prompt).content
        except Exception as e:
            print(f"[{i}] 调用失败：{e}")
            continue
        items = parse_json(out)
        got = 0
        for it in items:
            if it["instruction"] in seen:
                continue   # 问题重复，丢掉
            seen.add(it["instruction"])
            it["category"] = cat
            it["source"] = name
            dataset.append(it)
            per_cat[cat] = per_cat.get(cat, 0) + 1
            got += 1
        print(f"[{i}/{len(jobs)}] {cat} {name} 块{ci} 角度{ai} → +{got}，累计 {len(dataset)}  {per_cat}")
        if got == 0:
            print("     解析失败，模型原样输出：", out[:100].replace("\n", " "))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 完成：共 {len(dataset)} 条 → {OUT_PATH}")
    print(f"分类统计：{per_cat}")
    print(f"用时 {time.time() - t0:.1f} 秒")


if __name__ == "__main__":
    main()
