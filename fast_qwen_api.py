from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
import os
import json
import base64
import io
import uuid
import time
import threading
import asyncio

app = FastAPI()

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load the model and processor if not already loaded"""
    global model, processor
    
    if model is None or processor is None:
        print("Loading model and processor...")
        min_pixels = 256*28*28
        max_pixels = 512*28*28
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype="auto", 
            device_map="cuda:1"
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        print("Model and processor loaded successfully!")

# Define Pydantic models for request validation
class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Union[str, Dict[str, str]]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    model: str = "qwen2.5-vl-7b-instruct"
    max_tokens: int = 1280

@app.on_event("startup")
async def startup_event():
    # Load model on startup
    load_model()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """OpenAI-compatible /v1/chat/completions endpoint"""
    # Convert OpenAI format messages to Qwen format
    qwen_messages = []
    for msg in request.messages:
        role = msg.role
        content = msg.content
        print(content)
        # Handle different content formats
        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            qwen_content = []
            for item in content:
                if item.get('type') == 'text':
                    qwen_content.append({"type": "text", "text": item.get('text', '')})
                elif item.get('type') == 'image_url':
                    image_url = item.get('image_url', {})
                    if isinstance(image_url, str):
                        qwen_content.append({"type": "image", "image": image_url})
                    elif isinstance(image_url, dict) and 'url' in image_url:
                        # Handle base64 images
                        qwen_content.append({"type": "image", "image": image_url['url']})
            qwen_messages.append({"role": role, "content": qwen_content})
    
    # Prepare for inference
    text = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(qwen_messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # Handle streaming or normal response
    if request.stream:
        return await stream_response(inputs, request.max_tokens, request.model)
    else:
        return await normal_response(inputs, request.max_tokens, request.model)

async def normal_response(inputs, max_tokens, model_name):
    """Generate a normal (non-streaming) response"""
    # Run generation in a separate thread to not block the event loop
    def generate():
        with torch.no_grad():
            return model.generate(**inputs, max_new_tokens=max_tokens)
    
    # Run in a thread pool
    loop = asyncio.get_event_loop()
    generated_ids = await loop.run_in_executor(None, generate)
    
    # Process the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Format in OpenAI API response format
    response = {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": inputs.input_ids.shape[1],
            "completion_tokens": len(generated_ids_trimmed[0]),
            "total_tokens": inputs.input_ids.shape[1] + len(generated_ids_trimmed[0])
        }
    }
    
    return JSONResponse(content=response)

async def stream_response(inputs, max_tokens, model_name):
    """Generate a streaming response"""
    request_id = f"chatcmpl-{str(uuid.uuid4())}"
    
    # Create a streamer
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Create generation kwargs
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        streamer=streamer
    )
    
    # Start a separate thread for generation
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    async def generate():
        # Start the streaming response
        start_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(start_chunk)}\n\n"
        
        # Stream the generated text
        for text in streamer:
            if text:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": text,
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # End of the stream
        end_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("fast_qwen:app", host="0.0.0.0", port=8000, reload=False)
