# Qwen2.5-VL API 服务器

基于Flask的API服务器，提供与OpenAI兼容的Qwen2.5-VL视觉语言模型接口。

## 概述

该服务器创建了一个REST API，允许您通过与OpenAI兼容的接口与Qwen2.5-VL多模态模型进行交互。服务器支持流式和非流式响应，并可以处理与OpenAI API格式相同的文本和图像输入。

## 系统要求

- Python 3.8+
- Flask
- PyTorch
- Transformers
- 具有足够VRAM的CUDA兼容GPU（建议至少16GB）

### API端点

#### 聊天补全

`POST /v1/chat/completions`

此端点与OpenAI的聊天补全API兼容，便于与现有应用程序集成。

**请求格式：**

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个有帮助的助手。"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "这张图片里有什么？"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
          }
        }
      ]
    }
  ],
  "model": "qwen2.5-vl-7b-instruct",
  "max_tokens": 1280,
  "stream": false
}
```

**响应格式（非流式）：**

```json
{
  "id": "chatcmpl-",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "qwen2.5-vl-7b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "图片显示了一只猫坐在窗台上。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 52,
    "completion_tokens": 12,
    "total_tokens": 64
  }
}
```

### 流式响应

要接收流式响应，在请求中设置`"stream": true`。服务器将以OpenAI格式返回服务器发送事件流。

### 处理图像

图像可以以两种格式发送：

1. 作为带有数据URL的base64编码字符串：
   ```
   "image_url": {
     "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
   }
   ```

2. 作为URL（如果您的`process_vision_info`实现支持）：
   ```
   "image_url": {
     "url": "https://example.com/image.jpg"
   }
   ```

## 技术细节

### 模型加载

模型在第一次请求时才懒加载。如果服务器启动但未立即使用，这可以节省资源。模型配置使用CUDA设备1（`cuda:1`），您可以根据硬件配置更改此设置。

### 线程处理

对于流式响应，模型生成在单独的线程中进行，以避免阻塞主Flask线程，允许应用程序在结果可用时流式传输部分结果。

### 错误处理

实现了基本的错误处理，但在生产环境中，您应该增强更强大的错误报告和处理机制。

## 客户端示例

以下是与API交互的简单Python示例：

```python
import requests
import json
import base64
from pathlib import Path

# 服务器URL
url = "http://localhost:8000/v1/chat/completions"

# 读取并编码图像
image_path = Path("example.jpg")
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# 创建请求负载
payload = {
    "messages": [
        {
            "role": "system",
            "content": "你是一个能描述图像的有帮助的助手。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这张图片里有什么？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "model": "qwen2.5-vl-7b-instruct",
    "max_tokens": 1280
}

# 发送请求
response = requests.post(url, json=payload)

# 打印响应
print(json.dumps(response.json(), indent=2))
```


