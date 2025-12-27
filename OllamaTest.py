from ollama import Client

# 连接到本地Ollama服务（默认运行在11434端口）
client = Client(
    host='http://localhost:11434'  # 这是关键，连接本地
)

response = client.chat(
  model='qwen3-vl:4b',  # 使用你本地的模型名
  messages=[
    {
      'role': 'user',
      'content': 'What is in this image?',
      'images': ['/home/xcj/Desktop/1.png']  # 使用本地图片路径
    },
    { 
      'role': 'system',
      "content": "请始终使用简体中文进行回复。"
    },  # 系统指令设定语言
  ]
  # messages = [
  #   {
  #       "role": "user",
  #       "content": "Describe this image.",
  #       "images": "/home/xcj/Desktop/demo.jpeg",
  #   },

  #   { 
  #     'role': 'system',
  #     "content": "请始终使用简体中文进行回复。"
  #   },  # 系统指令设定语言
  #   ]
)
# prompt = """你是一个机器人，正在通过相机观察周围环境。你的任务是 识别并描述 你所看到的场景，重点关注 [障碍物、可行走区域、以及其他移动物体]。基于你的场景理解，决定你的下一步动作 (从 '前进', '后退', '左转', '右转', '停止' 中选择一个)，避免与障碍物距离过近。\n
# 用中文回答，具体固定输出格式如下：\n
# {"scene_description": "[口语化对场景的文字描述，**重点描述障碍物、可行走区域和移动物体**]","next_action": "[**从 '前进', '后退', '左转', '右转', '停止' 中选择一个动作**]"}
# """
print(response['message']['content'])
