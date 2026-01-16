import sys
import os
import asyncio
import time
import numpy as np
import cv2
from loguru import logger

sys.path.append('/dots.ocr')
try:
    from dots_ocr.model.layout_service import LayoutDetectionServiceONNX
except ImportError as e:
    logger.error(f"Import failed, please check the path. Current sys.path: {sys.path}")
    logger.error(f"Error details: {e}")
    sys.exit(1)

async def heartbeat_task():
    logger.info("❤️ Heartbeat monitoring started...")
    last_time = time.time()
    
    while True:
        await asyncio.sleep(0.5) 
        current_time = time.time()
        diff = current_time - last_time
        
        if diff > 3:
            logger.warning(f"❌ Heartbeat lagging significantly! Interval: {diff:.3f}s (Main thread blocked)")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❤️ Heartbeat normal (Interval: {diff:.3f}s)")
            
        last_time = current_time

async def main():
    MODEL_PATH = "/app/models/pp_doclayoutv2.onnx" 

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}, please modify the path in the test script.")
        return

    logger.info("Initializing LayoutDetectionServiceONNX...")
    try:
        service = LayoutDetectionServiceONNX(
            model_path=MODEL_PATH,
            concurrency=4,
            use_cpu=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    images = []
    test_image_dir = "/dots.ocr/ilovepdf_pages-to-jpg2"
    for filename in os.listdir(test_image_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(test_image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue
        images.append(img)

    heartbeat = asyncio.create_task(heartbeat_task())

    logger.info(">>> Calling _get_layout_image (Starting concurrent processing)...")
    start_time = time.time()

    try:
        results, inlines = await service._get_layout_image(images)
        
        end_time = time.time()
        logger.success(f"Processing complete! Total time: {end_time - start_time:.2f}s")
        logger.info(f"Number of results returned: {len(results)}")
        
        for i, res in enumerate(results):
            box_count = len(res.get('full_layout_info', []))
            print(f"  - Image {i}: Detected {box_count} boxes")

    except Exception as e:
        logger.exception("An error occurred during execution")
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(main())