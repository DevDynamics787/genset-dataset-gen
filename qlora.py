import argparse, gc, inspect, json, os, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

BASE_MODEL   = "/home/user/workspace/QwenPretrain"                # 底模
DATA_PATH    = "/home/user/workspace/model_b/data/dataset.json"   # 训练数据
TEMPLATE     = "/home/user/workspace/tmp.doc"                     # 提示词模板
ADAPTER_DIR  = "/home/user/workspace/model_b/checkpoint-lora"     # 中间产物：LoRA 适配器
OUT_DIR      = "/home/user/workspace/model_b/checkpoint-best"     # 最终交付：合并后的模型
MAX_LEN      = 512
LORA_R       = 8                       # 秩
LORA_ALPHA   = 16                      # 缩放 alpha
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # ★ qkvo 四个注意力层
LR           = 2e-4                    # 学习率
EPOCHS       = 3                       # 训练轮数
BATCH_SIZE   = 4                       # 单卡 batch
GRAD_ACCUM   = 4                       # 梯度累积（等效 batch = 4 × 4 = 16）
USE_4BIT     = True                    # 4bit 量化加载（nf4）


def load_template(path=TEMPLATE):
    """读 tmp.doc 模板 —— {instruction} 是占位符"""
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_dataset(tok):
    """把 dataset.json 按 tmp.doc 模板拼成训练文本"""
    tpl = load_template()
    items = json.load(open(DATA_PATH, encoding="utf-8"))
    rows = []
    for it in items:
        prompt = tpl.replace("{instruction}", it["instruction"])
        if it.get("input"):     # 本数据集 input 都是空的；非空时拼在指令后面
            prompt = prompt.replace("### 回答：", f"补充信息：{it['input']}\n### 回答：")
        # 末尾必须补 EOS！不然模型学会"永远不停"，推理时收不住
        rows.append({"text": prompt + it["output"] + tok.eos_token})
    print(f"训练样本：{len(rows)} 条")
    print("---- 拼出来的第一条长这样 ----")
    print(rows[0]["text"])
    print("------------------------------")
    return Dataset.from_list(rows)


def load_model_4bit():
    """4bit（nf4）量化加载底模 + 打开训练模式"""
    bnb = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type="nf4",                 # ★ nf4 量化
        bnb_4bit_compute_dtype=torch.bfloat16,     # 计算时用 bf16
        bnb_4bit_use_double_quant=True,            # 二次量化，更省显存
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False                                  # 开梯度检查点时必须关
    model = prepare_model_for_kbit_training(model)                  # 把 norm 层转 fp32、开梯度检查点
    print(f"4bit 加载完成，显存 {torch.cuda.memory_allocated()/1024**3:.2f} GB（LoRA 稍后由 Trainer 挂载）")
    return model


def train(model, tok, ds, dry_run=False):
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    kw = dict(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=1 if dry_run else EPOCHS,
        max_steps=1 if dry_run else -1,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )
    p = inspect.signature(SFTConfig.__init__).parameters
    kw["max_length" if "max_length" in p else "max_seq_length"] = MAX_LEN
    cfg = SFTConfig(**kw)

    tr_kw = dict(model=model, args=cfg, train_dataset=ds, peft_config=lora)
    tp = inspect.signature(SFTTrainer.__init__).parameters
    tr_kw["processing_class" if "processing_class" in tp else "tokenizer"] = tok

    trainer = SFTTrainer(**tr_kw)
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    print(f"LoRA 配置：r={LORA_R} alpha={LORA_ALPHA} target_modules={TARGET_MODULES}")
    print("开始训练…")
    trainer.train()
    trainer.save_model(ADAPTER_DIR)             # 先存 LoRA 适配器
    tok.save_pretrained(ADAPTER_DIR)
    print(f"LoRA 适配器已存到 {ADAPTER_DIR}")
    return trainer


def merge_and_save(tok):
    """把 LoRA 合并回底模 → checkpoint-best
       注意：必须用 bf16 的底模来合并，直接对 4bit 模型 merge 会得到垃圾权重"""
    print("开始合并（用 bf16 底模，不用 4bit 的）…")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model = model.merge_and_unload()          # 把 LoRA 权重加进底模
        model.config.use_cache = True
    except torch.cuda.OutOfMemoryError:
        print("显存不够，改用 CPU 合并…")
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model = model.merge_and_unload()
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    print(f"✅ 合并后的模型已存到 {OUT_DIR}（可用 from_pretrained 直接加载）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只跑 1 步，验证脚本")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds = build_dataset(tok)
    model = load_model_4bit()
    trainer = train(model, tok, ds, dry_run=args.dry_run)

    if args.dry_run:
        print("✅ dry-run 通过：脚本能跑通，4bit + LoRA 挂载正常")
        return

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    merge_and_save(tok)


if __name__ == "__main__":
    main()
