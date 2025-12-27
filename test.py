from modelscope import AutoModelForVision2Seq, AutoTokenizer
import torch

# 指定本地模型路径
local_model_path = "/home/xcj/work/qwen3/models/Qwen3-VL-2B-Instruct"

# 从本地加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,  # 明确指定精度，节省显存
    device_map="auto",
    trust_remote_code=True
)

# 准备模型输入（纯文本推理）
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 进行文本生成
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 解析生成的内容
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 解析thinking内容
try:
    if 151668 in output_ids:  # </think> token
        index = output_ids.index(151668)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index+1:], skip_special_tokens=True).strip("\n")
        print("thinking content:", thinking_content)
        print("content:", content)
    else:
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        print("content:", content)
except Exception as e:
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    print("content:", content)
    print("Note: Failed to parse thinking content:", e)
