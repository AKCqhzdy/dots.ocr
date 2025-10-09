import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from PIL import Image

from dots_ocr.utils.page_parser import PageParser


async def test_internVL():
    parser = PageParser()
    image = Image.open("/dots.ocr/charts/1.png")
    response = await parser._inference_with_vllm_internVL(
        image, "describe the picture in detail"
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(test_internVL())
