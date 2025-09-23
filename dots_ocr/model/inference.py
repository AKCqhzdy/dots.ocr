import json
import io
import base64
import math
from PIL import Image
import requests
from dots_ocr.utils.image_utils import PILimage_to_base64
from openai import AsyncOpenAI
import os
import httpx

async def inference_with_vllm(
        image,
        prompt, 
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='dotsocr',
        ):
    
    addr = f"http://{ip}:{port}/v1"
    client = AsyncOpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr, timeout=60)
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = await client.chat.completions.create(
            messages=messages, 
            model=model_name, 
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except httpx.TimeoutException:
        print("request timeout")
        return "timeout"
    except httpx.RequestError as e:
        print("request error:", e)
        return "error"

