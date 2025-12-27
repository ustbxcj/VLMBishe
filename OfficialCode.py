from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "/home/xcj/work/qwen3/models/Qwen3-VL-4B-Instruct", 
    # dtype="auto", 
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("/home/xcj/work/qwen3/models/Qwen3-VL-4B-Instruct")
#############
###Test1#####图片推断
#############
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/xcj/Desktop/1.png",
            },
            {
                "type": "text", 
                "text": "Describe this image.",
            },
        ],
    },
    {
        "role": "system",
        "content": [
            {
                "type": "text", 
                "text": "请始终使用简体中文进行回复"
            },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


#############
###Test2#####视频推测，这个运行不了，爆显存了
#############

# Messages containing a video url(or a local path) and a text query
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

# # Preparation for inference
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# )
# inputs = inputs.to(model.device)

# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

#############
###Test3#####视频采样推断，这个运行不了，爆显存了
#############
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
#             },
#             {"type": "text", "text": "Describe this video."},
#         ],
#     }
# ]

# # for video input, we can further control the fps or num_frames. \
# # defaultly, fps is set to 2

# # set fps = 4
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt",
#     fps=4
# )
# inputs = inputs.to(model.device)

# set num_frames = 128 and overwrite the fps to None!
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt",
#     num_frames=128,
#     fps=None,
# )
# inputs = inputs.to(model.device)

# Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)


# prompt = """你是一个机器人，正在通过相机观察周围环境。你的任务是 识别并描述 你所看到的场景，重点关注 [障碍物、可行走区域、以及其他移动物体]。基于你的场景理解，决定你的下一步动作 (从 '前进', '后退', '左转', '右转', '停止' 中选择一个)，避免与障碍物距离过近。\n
# 用中文回答，具体固定输出格式如下：\n
# {"scene_description": "[口语化对场景的文字描述，**重点描述障碍物、可行走区域和移动物体**]","next_action": "[**从 '前进', '后退', '左转', '右转', '停止' 中选择一个动作**]"}
# """