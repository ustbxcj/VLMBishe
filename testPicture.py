from modelscope import AutoModelForVision2Seq, AutoTokenizer, AutoProcessor
from PIL import Image
import torch

# 指定本地模型路径
local_model_path = "/home/xcj/work/qwen3/models/Qwen3-VL-2B-Instruct"

# 加载所有组件
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 准备图片
image_path = "/home/xcj/Desktop/1.png"
image = Image.open(image_path).convert("RGB")

# 构建对话
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# 使用processor处理输入
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(model.device)

# 生成
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# 解码结果
generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
response = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)[0]

print("Response:", response)
