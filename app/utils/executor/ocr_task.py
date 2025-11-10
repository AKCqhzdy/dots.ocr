import asyncio
import time
from pathlib import Path

import fitz
from loguru import logger
from opentelemetry import trace
from PIL import Image
from pydantic import BaseModel

from app.utils.executor.job_executor_pool import JobResponseModel
from app.utils.executor.stats import OcrTaskStats
from app.utils.executor.task_executor_pool import TaskExecutorPool
from app.utils.storage import StorageManager
from app.utils.tracing import get_tracer, start_child_span, traced
from dots_ocr.model.inference import InferenceTask, OcrInferenceTask
from dots_ocr.utils.page_parser import PageParser
from dots_ocr.utils.pdf_extractor import PdfExtractor

class OcrTaskModel(BaseModel):
    job_response: JobResponseModel
    task_id: str  # in case of pdf job, this is the page index

    output_file_name: str

    @property
    def original_file_uri(self):
        return self.job_response.input_s3_path

    @property
    def local_save_dir(self):
        job_files = self.job_response.get_job_local_files()
        return str(job_files.output_dir_path)


# TODO(tatiana): Make cpu part execute in parallel?
#                See if the cpu processing is blocking inference task scheduling.
class OcrTask:
    def __init__(
        self,
        span: trace.Span,
        task_model: OcrTaskModel,
        parser: PageParser,
        ocr_inference_pool: TaskExecutorPool,
        describe_picture_pool: TaskExecutorPool,
    ):
        self._span = span
        self._task_model = task_model
        self._stats = OcrTaskStats(status="pending")
        self._ocr_inference_pool = ocr_inference_pool
        self._describe_picture_pool = describe_picture_pool
        self._parser = parser
        self._page_index = 0

    @property
    def job_id(self):
        return self._task_model.job_response.job_id

    @property
    def task_id(self):
        return self._task_model.task_id

    @property
    def describe_picture(self):
        return self._task_model.job_response.describe_picture

    @property
    def prompt_mode(self):
        return self._task_model.job_response.prompt_mode

    @property
    def status(self):
        return self._stats.status

    @property
    def error_msg(self):
        return self._stats.error_msg

    @property
    def token_usage(self):
        return self._stats.token_usage

    @traced()
    async def _submit_ocr_inference_task(self, task_id, image, prompt):
        task = OcrInferenceTask(
            start_child_span(f"OcrInferenceTask {task_id}"),
            self._parser.ocr_inference_task_options,
            task_id,
            image,
            prompt,
        )
        await self._ocr_inference_pool.add_task(task)
        return task.get_completion_future(), task

    @traced()
    async def _submit_describe_picture_task(self, task_id, image, prompt):
        task = InferenceTask(
            start_child_span(f"DescribePictureTask {task_id}"),
            self._parser.describe_picture_task_options,
            task_id,
            image,
            prompt,
        )
        await self._describe_picture_pool.add_task(task)
        return task.get_completion_future(), task

    @traced()
    async def _process_ocr_results(
        self, ocr_results, origin_image, image, scale_factor, toc=[]
    ):
        self._span.add_event("start process ocr results")
        start_time = time.perf_counter()
        try:
            cells = await self._parser.process_results(
                (
                    self._task_model.local_save_dir
                    if not self.describe_picture
                    else None
                ),
                (
                    self._task_model.output_file_name
                    if not self.describe_picture
                    else None
                ),
                ocr_results,
                self.prompt_mode,
                origin_image,
                image,
                self._page_index,
                scale_factor,
                toc,
            )
        except Exception:
            logger.exception(
                f"Error processing results for page {self._page_index} "
                f"of {self._task_model.original_file_uri}"
            )
            raise


        end_time = time.perf_counter()
        elapsed = end_time - start_time
        self._span.add_event("end process ocr results")
        self._span.set_attribute("post_process_ocr_results_wall_time_s", elapsed)

        return cells

    def iter_picture_blocks(self, cells: dict, origin_image: Image.Image):
        for info_block in cells["full_layout_info"]:
            if info_block["category"] == "Picture":
                x0, y0, x1, y1 = info_block["bbox"]
                cropped_img = origin_image.crop((x0, y0, x1, y1))
                yield info_block, cropped_img
    @traced()
    async def _describe_pictures_in_page(self, cells: dict, origin_image: Image.Image):
        prompt = self._parser.picture_description_prompt
        futures: list[asyncio.Future] = []
        picture_blocks: list[dict] = []
        try:
            idx = 0
            tasks: list[InferenceTask] = []
            for picture_block, cropped_img in self.iter_picture_blocks(
                cells, origin_image
            ):
                future, task = await self._submit_describe_picture_task(
                    f"{self.job_id}-{self._page_index}-describe-{idx}",
                    cropped_img,
                    prompt,
                )
                idx += 1
                futures.append(future)
                tasks.append(task)
                picture_blocks.append(picture_block)

            await asyncio.gather(*futures)

        except Exception as e:
            logger.error(f"Error submitting picture description inference task: {e}")
            raise

        for picture_block, future, task in zip(picture_blocks, futures, tasks):
            if future.cancelled():
                logger.warning(
                    "Future for picture description was cancelled for "
                    f"page {self._page_index} of doc {self._task_model.original_file_uri}",
                )
                break
            if future.exception():
                raise RuntimeError(
                    "Failed to get description of picture block(s) for "
                    f"page {self._page_index} of doc {self._task_model.original_file_uri}"
                ) from future.exception()
            # is_fallback only in last attempt it will be set true by_fallback_ocr
            if task.is_timeout:
                self._stats.status = "timeout"
            elif task.is_fallback:
                self._stats.status = "fallback"
            if task.success_usage:
                self._stats.add_token_usage(*task.success_usage)
            picture_block["text"] = future.result().strip()

    def final_success(self):
        self._span.end()

    def final_failure(self, error: str):
        self._span.set_status(trace.Status(trace.StatusCode.ERROR, error))
        self._span.end()

    async def run(self):
        self._stats.status = "running"
        with trace.use_span(self._span, end_on_exit=False):
            with get_tracer().start_as_current_span(
                f"attempt-{self._stats.attempt}"
            ) as span:
                try:
                    self._stats.attempt += 1
                    start_time = time.perf_counter()
                    logger.debug(
                        f"Start processing page {self._page_index} of "
                        f"doc {self._task_model.original_file_uri} (attempt {self._stats.attempt})"
                    )
                    result = await self._run()
                    self._stats.task_execution_time = time.perf_counter() - start_time
                    if self._stats.status != "fallback":
                        self._stats.status = "finished"
                    span.set_attribute(
                        "task_execution_wall_time_s", self._stats.task_execution_time
                    )
                    span.set_attribute("task_status", self._stats.status)
                    span.add_event("Finished the OCR task")
                    return result
                except Exception as e:
                    logger.error(f"Error running OCR task: {e}", exc_info=True)
                    if self._stats.status != "timeout":
                        self._stats.status = "failed"
                    self._stats.error_msg = str(e)
                    span.set_attribute("task_status", self._stats.status)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    return None

    async def _run(self):
        pass


class PdfOcrTask(OcrTask):
    """A CPU-bound task."""

    def __init__(self, page: fitz.Page, storage_manager: StorageManager, toc: dict, **kwargs):
        super().__init__(**kwargs)
        self._page_index = int(self._task_model.task_id)
        self._page = page
        self._storage_manager = storage_manager
        self._toc = toc

    @traced()
    async def _upload_results(self, result: dict):
        page_upload_tasks = []
        paths_to_upload = {
            "md": result.get("md_content_path"),
            "md_nohf": result.get("md_content_nohf_path"),
            "json": result.get("layout_info_path"),
        }

        job_files = self._task_model.job_response.get_job_local_files()
        for _, local_path in paths_to_upload.items():
            if local_path:
                file_name = Path(local_path).name
                s3_key = f"{job_files.remote_output_file_key}/{file_name}"
                task = asyncio.create_task(
                    self._storage_manager.upload_file(
                        job_files.remote_output_bucket,
                        s3_key,
                        local_path,
                        self._task_model.job_response.is_s3,
                    )
                )
                page_upload_tasks.append(task)
        await asyncio.gather(*page_upload_tasks)
        paths_to_upload["page_no"] = self._page_index
        return paths_to_upload

    @traced()
    async def _run(self):
        """
        Returns:
            dict: keys are "md", "md_nohf", "json", "page_no"
        """
        try:
            origin_image, image, prompt, scale_factor = self._parser.prepare_pdf_page(
                self._page, self.prompt_mode, bbox=None
            )
            # transform toc coordinates from pdf space to image space
            logger.debug(f"Page index: {self._page_index}, TOC: {self._toc}")
            if self._toc is not None:
                for entry in self._toc:
                    entry["to"][0] = entry["to"][0] * scale_factor
                    entry["to"][1] = entry["to"][1] * scale_factor
        except Exception as e:
            logger.error(f"Error preparing image and prompt: {e}")
            raise

        # If the inference task queue is full, this coroutine will block
        try:
            inference_future, inference_task = await self._submit_ocr_inference_task(
                f"{self.job_id}-{self._page_index}-ocr", image, prompt
            )
        except Exception as e:
            logger.error(f"Error submitting OCR inference task: {e}")
            raise

        try:
            inference_result = await inference_future
            if inference_task.is_fallback:
                self._stats.status = "fallback"
            if inference_task.success_usage:
                self._stats.add_token_usage(*inference_task.success_usage)
        except Exception as e:
            if inference_task.is_timeout:
                self._stats.status = "timeout"
            logger.error(f"Error getting OCR inference result: {e}")
            raise

        try:
            cells = await self._process_ocr_results(
                inference_result, origin_image, image, scale_factor, self._toc
            )
        except Exception as e:
            logger.error(f"Error post-processing ocr results: {e}")
            raise

        if self.describe_picture:
            try:
                await self._describe_pictures_in_page(cells, origin_image=image)
            except Exception as e:
                logger.error(
                    f"Error describing pictures in page {self._page_index}: {e}"
                )
                raise

            try:
                cells = await self._parser.save_results(
                    cells,
                    self._task_model.local_save_dir,
                    self._task_model.output_file_name,
                    image,
                    scale_factor,
                )
                logger.debug(
                    f"Saved results for page {self._page_index} "
                    f"of doc {self._task_model.original_file_uri}: {cells}"
                )
            except Exception as e:
                logger.error(
                    f"Error saving results for page {self._page_index} "
                    f"of doc {self._task_model.original_file_uri}: {e}"
                )
                raise

        return await self._upload_results(cells)


class ImageOcrTask(OcrTask):
    def __init__(self, input_path: str, bbox: tuple = None, **kwargs):
        super().__init__(**kwargs)
        self._input_path = input_path
        self._bbox = bbox

    @traced()
    async def _run(self):
        origin_image, image, prompt, scale_factor = (
            await self._parser.prepare_image_for_ocr(
                self._input_path,
                self.prompt_mode,
                self._task_model.job_response.fitz_preprocess,
                self._bbox,
            )
        )

        try:
            ocr_future, ocr_task = await self._submit_ocr_inference_task(
                f"{self.job_id}-ocr", image, prompt
            )
        except Exception as e:
            logger.error(f"Error submitting OCR inference task: {e}")
            raise

        try:
            ocr_result = await ocr_future
            if ocr_task.is_fallback:
                self._stats.status = "fallback"
            if ocr_task.success_usage:
                self._stats.add_token_usage(*ocr_task.success_usage)
        except Exception as e:
            logger.error(f"Error getting OCR inference result: {e}")
            raise

        try:
            cells = await self._process_ocr_results(
                ocr_result, origin_image, image, scale_factor
            )
        except Exception as e:
            logger.error(f"Error post-processing ocr results: {e}")
            raise

        if self.describe_picture:
            try:
                await self._describe_pictures_in_page(cells, origin_image=image)
            except Exception as e:
                logger.error(
                    f"Error describing pictures in image {self._task_model.original_file_uri}: {e}"
                )
                raise

            try:
                cells = await self._parser.save_results(
                    cells,
                    self._task_model.local_save_dir,
                    self._task_model.output_file_name,
                    origin_image,
                    1,
                )
                logger.debug(
                    f"Saved results for image {self._task_model.original_file_uri}: {cells}"
                )
            except Exception as e:
                logger.error(
                    f"Error saving results for image {self._task_model.original_file_uri}: {e}"
                )
                raise

        return cells




class PipeOcrTask(OcrTask):
    """A CPU-bound task."""

    def __init__(
        self,
        page: fitz.Page,
        storage_manager: StorageManager, 
        pdf_extractor: PdfExtractor,
        **kwargs):
        super().__init__(**kwargs)
        self._page_index = int(self._task_model.task_id)
        self._page = page
        self._storage_manager = storage_manager
        self._pdf_extractor = pdf_extractor
        
    @traced()
    async def _upload_results(self, result: dict):
        page_upload_tasks = []
        paths_to_upload = {
            "md": result.get("md_content_path"),
            "md_nohf": result.get("md_content_nohf_path"),
            "json": result.get("layout_info_path"),
        }

        job_files = self._task_model.job_response.get_job_local_files()
        for _, local_path in paths_to_upload.items():
            if local_path:
                file_name = Path(local_path).name
                s3_key = f"{job_files.remote_output_file_key}/{file_name}"
                task = asyncio.create_task(
                    self._storage_manager.upload_file(
                        job_files.remote_output_bucket,
                        s3_key,
                        local_path,
                        self._task_model.job_response.is_s3,
                    )
                )
                page_upload_tasks.append(task)
        await asyncio.gather(*page_upload_tasks)
        paths_to_upload["page_no"] = self._page_index
        return paths_to_upload

    @traced()
    async def _run(self):
        """
        Returns:
            dict: keys are "md", "md_nohf", "json", "page_no"
        """
        origin_image, scale_factor = self._pdf_extractor.page_to_image(self._page_index, self._parser.dpi)
        try:
            cells = await self._parser._layout_detection(origin_image)
            cells['page_no'] = self._page_index
            if logger._core.min_level <= logger.level("DEBUG").no:
                i=0
                for info_block in cells["full_layout_info"]:
                    i+=1
                    self._pdf_extractor.crop_bbox(self._page_index, info_block['bbox'], f"/dots.ocr/test/outputs/crop/{self._page_index}/crop_{i}.png" ,self._parser.dpi)
                logger.debug(f"Layout detection results: {cells}")
        except Exception as e:
            logger.error(f"Error while detecting layout: {e}")
            raise
        
        try:
            # TODO(zihao): need align category between layout detection model and dots
            for info_block in cells["full_layout_info"]:
                if (
                    info_block["category"] == "table"
                    or info_block["category"] == "figure"
                ):
                    pass
                else:
                    block_in_pdf_size = [ i / scale_factor for i in info_block["bbox"] ]
                    info_block["text"] = self._pdf_extractor.extract_text_from_page(
                        self._page_index, block_in_pdf_size
                    )
            logger.debug(f"Extracted text results: {cells}")
        except Exception as e:
            logger.error(f"Error while extracting: {e}")
            raise

        try:
            await self._parser._layout_reader(cells["full_layout_info"], cells['width'], cells['height'])
        except Exception as e:
            logger.error(f"Error while sorting blocks: {e}")
            raise
        
        
        if self.describe_picture:
            try:
                await self._describe_pictures_in_page(cells, origin_image=origin_image)
            except Exception as e:
                logger.error(
                    f"Error describing pictures in page {self._page_index}: {e}"
                )
                raise

        try:
            cells = await self._parser.save_results(
                cells,
                self._task_model.local_save_dir,
                self._task_model.output_file_name,
                origin_image,
                scale_factor,
            )
            logger.debug(
                f"Saved results for page {self._page_index} "
                f"of doc {self._task_model.original_file_uri}: {cells}"
            )
        except Exception as e:
            logger.error(
                f"Error saving results for page {self._page_index} "
                f"of doc {self._task_model.original_file_uri}: {e}"
            )
            raise

        return await self._upload_results(cells)