import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 路径配置（用绝对路径） =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/lora
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))      # 项目根

BASE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "src", "models", "Qwen", "Qwen2.5-1.5B-Instruct"
)
LORA_PATH = os.path.join(BASE_DIR, "qwen_lora_output")
OUTPUT_PATH = os.path.join(BASE_DIR, "qwen_merged")

print("基础模型路径:", BASE_MODEL_PATH)
print("LoRA 路径:", LORA_PATH)
print("输出路径:", OUTPUT_PATH)

# 检查路径是否存在
assert os.path.isdir(BASE_MODEL_PATH), f"基础模型路径不存在: {BASE_MODEL_PATH}"
assert os.path.isdir(LORA_PATH), f"LoRA 路径不存在: {LORA_PATH}"

# ================= 加载基础模型 =================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)

# ================= 加载 LoRA 适配器 =================
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# ================= 融合 =================
model = model.merge_and_unload()

# ================= 保存 =================
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ 融合完成！")

# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MERGED_PATH = os.path.join(BASE_DIR, "qwen_merged")
#
# tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MERGED_PATH, dtype=torch.float16, device_map="cuda", trust_remote_code=True
# )
# model.eval()
#
# def predict(user_input):
#     instruction = "你是一个对话意图分类器。判断用户输入的意图类别，只输出chat或者rag：\n- chat：闲聊、问候、感谢、道别、情感表达、天气如何等与知识查询无关的对话\n 结合对话历史判断指代不明确的输入（如\"那第二个呢\"）\n1. 只要涉及事实性知识诉求，一律归为 rag\n2. 无法确定时输出 rag"
#     messages = [{"role": "user", "content": f"{instruction}\n\n{user_input}"}]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=10,
#             do_sample=False,
#             pad_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#         )
#     gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
#     return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
#
# print(predict("今天会议内容大纲帮我总结"))