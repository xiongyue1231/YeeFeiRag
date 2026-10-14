import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 路径配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/lora
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))      # 项目根

MODEL_NAME = os.path.join(
    PROJECT_ROOT, "src", "models", "Qwen", "Qwen2.5-1.5B-Instruct"
)
LORA_PATH = os.path.join(BASE_DIR, "qwen_lora_output")

print("基础模型:", MODEL_NAME)
print("LoRA 适配器:", LORA_PATH)

# ================= 加载模型 =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,     # Windows 建议 fp16
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("模型加载完成\n")

# ================= 推理函数 =================
def predict(user_input):
    instruction = "你是一个对话意图分类器。判断用户输入的意图类别，只输出chat或者rag：\n- chat：闲聊、问候、感谢、道别、情感表达、天气如何等与知识查询无关的对话\n 结合对话历史判断指代不明确的输入（如\"那第二个呢\"）\n1. 只要涉及事实性知识诉求，一律归为 rag\n2. 无法确定时输出 rag"

    messages = [
        {"role": "user", "content": f"{instruction}\n\n{user_input}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,          # 贪心解码，保证稳定
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return result


# ================= 测试样本 =================
test_cases = [
    # (输入, 期望标签)
    ("你好呀，今天过得怎么样？", "chat"),
    ("给我讲个笑话吧", "chat"),
    ("我心情不太好，能安慰我一下吗", "chat"),
    ("谢谢你的帮助，你真好", "chat"),
    ("帮我写一首关于春天的诗", "chat"),
    ("你觉得人工智能未来会取代人类吗？", "chat"),
    ("我们公司今年的年假政策是怎样的？", "rag"),
    ("产品 X200 的最大承重是多少？", "rag"),
    ("入职需要准备哪些材料？", "rag"),
    ("报销流程是什么样的？", "rag"),
    ("这份合同的违约金条款是怎么规定的？", "rag"),
    ("售后服务热线是多少？", "rag"),
    # 边界/困难样本
    ("你好，我想问下年假政策", "rag"),
    ("今天天气怎么样？", "chat"),
    ("你们公司今天上班吗？", "chat"),
    ("1+1等于几？", "chat"),
    ("本周会议大纲？", "rag"),
]

# ================= 逐条测试 =================
correct = 0
total = len(test_cases)
print(test_cases)
print(f"{'输入':<30} {'期望':<10} {'预测':<10} {'结果'}")
print("-" * 70)

for text, expected in test_cases:
    pred = predict(text)
    # 有些输出可能带标点或多余字符，做一次简单清洗
    pred_clean = pred.lower().replace(".", "").replace("。", "").strip()
    print(pred_clean)
    if "chat" in pred_clean:
        pred_clean = "chat"
    elif "rag" in pred_clean:
        pred_clean = "rag"

    ok = "✅" if pred_clean == expected else "❌"
    if pred_clean == expected:
        correct += 1

    print(f"{text:<30} {expected:<10} {pred_clean:<10} {ok}")

print("-" * 70)
print(f"准确率: {correct}/{total} = {correct / total * 100:.1f}%")