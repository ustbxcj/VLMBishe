from ollama import Client

# 连接到本地Ollama服务（默认运行在11434端口）
client = Client(
    host='http://localhost:11434'  # 这是关键，连接本地
)
prompt1 = 'What is in this image?'
prompt2 = "请始终使用简体中文进行回复。"
image_path = '/home/xcj/Desktop/1.png'
response = client.chat(
  model='qwen3-vl:4b',  # 使用你本地的模型名
  messages=[
    {
      'role': 'user',
      'content': prompt1,
      'images': [image_path]  # 使用本地图片路径
    },
    { 
      'role': 'system',
      "content": prompt2
    },  # 系统指令设定语言
  ]
)
print(response['message']['content'])
