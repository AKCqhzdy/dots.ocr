import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fitz import Page
from PIL import Image
from pydantic import BaseModel

from app.utils.tracing import start_child_span, traced
from dots_ocr.model.inference import (
    InferenceTask,
    InferenceTaskOptions,
    OcrInferenceTask,
)
from dots_ocr.utils.consts import MAX_PIXELS, MIN_PIXELS
from dots_ocr.utils.doc_utils import fitz_doc_to_image
from dots_ocr.utils.format_transformer import layoutjson2md
from dots_ocr.utils.image_utils import fetch_image, get_image_by_fitz_doc
from dots_ocr.utils.layout_utils import (
    draw_layout_on_image,
    post_process_output,
    pre_process_bboxes,
)
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.model.layout_service import get_layout_image, sort_bboxes



class ParseOptions(BaseModel):
    dpi: int = 200
    min_pixels: int = None
    max_pixels: int = None
    task_retry_count: int = 3


class PageParser:
    """
    Asynchronous parser for image or PDF files.
    """

    def __init__(
        self,
        ocr_inference_task_options: InferenceTaskOptions = None,
        describe_picture_task_options: InferenceTaskOptions = None,
        parse_options: ParseOptions = None,
        concurrency_limit=8,
    ):
        assert (
            parse_options.min_pixels is None or parse_options.min_pixels >= MIN_PIXELS
        )
        assert (
            parse_options.max_pixels is None or parse_options.max_pixels <= MAX_PIXELS
        )
        self._image_options = parse_options
        if self._image_options is None:
            self._image_options = ParseOptions()

        self._ocr_inference_task_options = ocr_inference_task_options
        self._describe_picture_task_options = describe_picture_task_options
        if self._ocr_inference_task_options is None:
            self._ocr_inference_task_options = InferenceTaskOptions(
                model_name="dotsocr",
                model_host="localhost",
                model_port=8000,
                temperature=0.1,
                top_p=1.0,
                max_completion_tokens=32768,
                timeout=10,
            )
        if self._describe_picture_task_options is None:
            self._describe_picture_task_options = InferenceTaskOptions(
                model_name="InternVL3_5-2B",
                model_host="internvl3-5",
                model_port=6008,
                temperature=0.1,
                top_p=1.0,
                max_completion_tokens=8192,
                timeout=10,
            )

        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        self.semaphore_reader = asyncio.Semaphore(self.concurrency_limit)  # TODO(zihao) can larger. need meansure later
        self.cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    @property
    def page_retry_number(self):
        return self._image_options.task_retry_count

    @property
    def ocr_inference_task_options(self):
        return self._ocr_inference_task_options

    @property
    def describe_picture_task_options(self):
        return self._describe_picture_task_options

    @property
    def dpi(self):
        return self._image_options.dpi

    @property
    def min_pixels(self):
        return self._image_options.min_pixels

    @property
    def max_pixels(self):
        return self._image_options.max_pixels

    @property
    def picture_description_prompt(self) -> str:
        return (
            "Extract the information from this image objectively and concisely. "
            "Do not add any extra explanation, commentary, or surrounding text. "
            "Detect the main content type in the image and follow the corresponding rule exactly:\n\n"
            "1. If the image contains one or more tables or charts/graphs:\n"
            "   - Output ONLY clean markdown tables that faithfully reproduce all data from the image.\n"
            "   - Use accurate headers and all rows/values exactly as shown.\n"
            "   - If there are multiple tables/charts, output each as a separate markdown table with a single empty line in between.\n"
            "   - Do not number them or add any titles unless they are explicitly present in the image.\n\n"
            "2. If the image primarily contains mathematical formulas/equations:\n"
            "   - Output ONLY the formulas in proper LaTeX format.\n"
            "   - Use display math mode ($$...$$) for standalone equations and inline $...$ where appropriate.\n"
            "   - If there are multiple equations, separate them with a single empty line.\n\n"
            "3. If the image is neither:\n"
            "   - Summarize it in three sentences. \n\n"
            "Never add phrases like 'Here is the extracted information', 'Table:', 'Formula:', or any other wrapper text."
        )

    def prepare_image(
        self,
        origin_image: Image.Image,
        source: Literal["image", "pdf"],
        fitz_preprocess=False,
    ):
        """Synchronous, CPU-bound part of image preparation."""
        scale_factor = 1.0
        if source == "image" and fitz_preprocess:
            image, scale_factor = get_image_by_fitz_doc(
                origin_image, target_dpi=self.dpi
            )
            image = fetch_image(
                image, min_pixels=self.min_pixels, max_pixels=self.max_pixels
            )
            return image, scale_factor
        image = fetch_image(
            origin_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        return image, scale_factor

    def prepare_ocr_prompt(
        self, origin_image: Image.Image, image: Image.Image, prompt_mode: str, bbox=None
    ):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == "prompt_grounding_ocr":
            assert bbox is not None
            bbox = pre_process_bboxes(
                origin_image, [bbox], input_width=image.width, input_height=image.height
            )[0]
            prompt += str(bbox)
        return prompt

    def prepare_image_and_prompt(
        self, origin_image, prompt_mode, source, fitz_preprocess=False, bbox=None
    ):
        """Synchronous, CPU-bound part of image preparation."""
        image, scale_factor = self.prepare_image(origin_image, source, fitz_preprocess)
        prompt = self.prepare_ocr_prompt(origin_image, image, prompt_mode, bbox)

        return (
            image,
            prompt,
            scale_factor,
        )  # only image with fitz_preprocess will receive scale_factor for use

    def _process_and_save_results(
        self,
        response,
        prompt_mode,
        save_dir,
        save_name,
        origin_image,
        image,
        scale_factor=1.0,
        toc=[],
    ):
        """Synchronous, CPU/IO-bound part of post-processing and saving."""
        os.makedirs(save_dir, exist_ok=True)
        result = {}
        cells, _ = post_process_output(response, prompt_mode, origin_image, image, toc)
        for cell in cells:
            cell["bbox"] = [int(float(num) / scale_factor) for num in cell["bbox"]]
        width, height = origin_image.size
        cells_with_size = {
            "width": int(float(width) / scale_factor),
            "height": int(float(height) / scale_factor),
            "full_layout_info": cells,
        }

        # try:
        #     draw_layout_on_image(origin_image, cells)
        # except Exception as e:
        #     print(f"Error drawing layout on image: {e}")

        json_path = os.path.join(save_dir, f"{save_name}.json")
        save_json = [cells_with_size]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_json, f, ensure_ascii=False, indent=4)
        result["layout_info_path"] = json_path

        md_content = layoutjson2md(origin_image, cells, text_key="text")
        md_content_nohf = layoutjson2md(
            origin_image, cells, text_key="text", no_page_hf=True
        )

        md_path = os.path.join(save_dir, f"{save_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        result["md_content_path"] = md_path

        md_nohf_path = os.path.join(save_dir, f"{save_name}_nohf.md")
        with open(md_nohf_path, "w", encoding="utf-8") as f:
            f.write(md_content_nohf)
        result["md_content_nohf_path"] = md_nohf_path

        return result

    def _process_results(
        self, response, prompt_mode, origin_image, image, page_idx=None, toc=[]
    ):
        """Synchronous, CPU/IO-bound part of post-processing and saving."""
        cells, _ = post_process_output(response, prompt_mode, origin_image, image, toc=toc)
        width, height = origin_image.size
        cells_with_size = {"width": width, "height": height, "full_layout_info": cells}
        if page_idx is not None:
            cells_with_size["page_no"] = page_idx
        return cells_with_size

    @traced()
    async def save_results(
        self, cells_with_size, save_dir, save_name, image_origin, scale_factor=1.0
    ):

        md_content = layoutjson2md(
            image_origin, cells_with_size["full_layout_info"], text_key="text"
        )
        md_content_nohf = layoutjson2md(
            image_origin,
            cells_with_size["full_layout_info"],
            text_key="text",
            no_page_hf=True,
        )

        for cell in cells_with_size["full_layout_info"]:
            cell["bbox"] = [int(float(num) / scale_factor) for num in cell["bbox"]]
        cells_with_size["width"] = int(float(cells_with_size["width"]) / scale_factor)
        cells_with_size["height"] = int(float(cells_with_size["height"]) / scale_factor)

        result = {}
        if cells_with_size["page_no"] is not None:
            result["page_no"] = cells_with_size["page_no"]
        save_json = [cells_with_size]
        json_path = os.path.join(save_dir, f"{save_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_json, f, ensure_ascii=False, indent=4)
        result["layout_info_path"] = json_path

        md_path = os.path.join(save_dir, f"{save_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        result["md_content_path"] = md_path

        md_nohf_path = os.path.join(save_dir, f"{save_name}_nohf.md")
        with open(md_nohf_path, "w", encoding="utf-8") as f:
            f.write(md_content_nohf)
        result["md_content_nohf_path"] = md_nohf_path

        return result

    # It is now deprecated
    async def _inference_with_vllm(self, image, prompt):
        task = OcrInferenceTask(
            start_child_span("OcrInferenceTask", None),
            self._ocr_inference_task_options,
            "ocr_inference_task",
            image,
            prompt,
        )
        return await task.inference_with_vllm()

    async def _inference_with_vllm_intern_vl(self, image, prompt):
        task = InferenceTask(
            start_child_span("InferenceTask", None),
            self._describe_picture_task_options,
            "describe_picture_task",
            image,
            prompt,
        )
        return await task.inference_with_vllm()

    # It is now deprecated
    async def _parse_single_image(
        self,
        origin_image,
        prompt_mode,
        save_dir,
        save_name,
        source="image",
        page_idx=0,
        bbox=None,
        fitz_preprocess=False,
        scale_factor=1.0,
    ):
        """Asynchronous pipeline for a single image."""
        async with self.semaphore:
            loop = asyncio.get_running_loop()

            # 1. Run CPU-bound image prep in executor
            image, prompt, _ = await loop.run_in_executor(
                self.cpu_executor,
                self.prepare_image_and_prompt,
                origin_image,
                prompt_mode,
                source,
                fitz_preprocess,
                bbox,
            )
            # 2. Make non-blocking network call for inference
            response = await self._inference_with_vllm(image, prompt)

            # 3. Run CPU/IO-bound post-processing and saving in executor
            cells = await self.process_results(
                save_dir,
                (f"{save_name}_page_{page_idx}" if source == "pdf" else save_name),
                response,
                prompt_mode,
                origin_image,
                image,
                page_idx,
                scale_factor,
            )
            return cells

    async def process_results(
        self,
        save_dir,
        save_name,
        response,
        prompt_mode,
        origin_image,
        image,
        page_idx,
        scale_factor,
        toc,
    ):
        loop = asyncio.get_running_loop()
        if save_dir is None:  # do not save, just return cells for further processing
            return await loop.run_in_executor(
                self.cpu_executor,
                self._process_results,
                response,
                prompt_mode,
                origin_image,
                image,
                page_idx,
                toc,
            )

        result = await loop.run_in_executor(
            self.cpu_executor,
            self._process_and_save_results,
            response,
            prompt_mode,
            save_dir,
            save_name,
            origin_image,
            image,
            scale_factor,
            toc,
        )
        result["page_no"] = page_idx
        return result

    async def _describe_picture_in_single_page(self, origin_image, cells):
        picture_blocks = [
            info_block
            for info_block in cells["full_layout_info"]
            if info_block["category"] == "Picture"
        ]
        # print(picture_blocks)

        if not picture_blocks:
            return

        # Create tasks for concurrent processing
        async def process_picture_block(info_block):
            async with self.semaphore:  # Use the existing semaphore from PageParser
                x0, y0, x1, y1 = info_block["bbox"]
                cropped_img = origin_image.crop((x0, y0, x1, y1))
                response = await self._inference_with_vllm_intern_vl(
                    cropped_img, self.picture_description_prompt
                )
                info_block["text"] = response.strip()

        # Process all pictures concurrently
        tasks = [process_picture_block(block) for block in picture_blocks]
        await asyncio.gather(*tasks)

    def _prepare_image_for_ocr(
        self,
        input_path: str,
        prompt_mode: str,
        fitz_preprocess=False,
        bbox=None,
        source="image",
    ):
        origin_image = fetch_image(input_path)
        image, scale_factor = self.prepare_image(origin_image, source, fitz_preprocess)
        prompt = self.prepare_ocr_prompt(origin_image, image, prompt_mode, bbox)
        return origin_image, image, prompt, scale_factor

    async def prepare_image_for_ocr(
        self,
        input_path: str,
        prompt_mode: str,
        fitz_preprocess=False,
        bbox=None,
    ):
        """Wrapped in thread pool due to blocking IO operations."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.cpu_executor,
            self._prepare_image_for_ocr,
            input_path,
            prompt_mode,
            fitz_preprocess,
            bbox,
            "image",
        )

    @traced()
    def prepare_pdf_page(self, page: Page, prompt_mode: str, bbox=None):
        """Synchronous, CPU-bound part of image preparation."""
        origin_image, scale_factor = fitz_doc_to_image(page, target_dpi=self.dpi)
        image, _ = self.prepare_image(origin_image, "pdf")
        prompt = self.prepare_ocr_prompt(origin_image, image, prompt_mode, bbox=bbox)

        return origin_image, image, prompt, scale_factor

