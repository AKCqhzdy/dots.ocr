import asyncio
import base64
from io import BytesIO

import fitz
from loguru import logger
from PIL import Image
from tqdm.asyncio import tqdm

from app.utils.executor.job_executor_pool import JobResponseModel
from app.utils.executor.ocr_task import ImageOcrTask, OcrTaskModel, PdfOcrTask
from app.utils.executor.task_executor_pool import TaskExecutorPool
from app.utils.storage import StorageManager
from dots_ocr.utils.directory_cleaner import DirectoryCleaner
from dots_ocr.utils.doc_utils import (
    load_images_from_pdf,
)
from dots_ocr.utils.page_parser import PageParser


def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class DotsOCRParser:
    def __init__(
        self,
        ocr_task_executor_pool: TaskExecutorPool,
        describe_picture_task_executor_pool: TaskExecutorPool,
        page_parser: PageParser,
        storage_manager: StorageManager,
    ):
        self.parser = page_parser
        self.directory_cleaner = None
        self._ocr_task_executor_pool = ocr_task_executor_pool
        self._describe_picture_task_executor_pool = describe_picture_task_executor_pool
        self._storage_manager = storage_manager

    async def parse_image(
        self,
        job_response: JobResponseModel,
        bbox=None,
    ):
        job_files = job_response.get_job_local_files()
        job_response.task_stats.total_task_count = 1
        task_model = OcrTaskModel(
            job_response=job_response,
            task_id=str(0),
            output_file_name=job_files.output_file_name,
        )
        task = ImageOcrTask(
            input_path=str(job_files.input_file_path),
            bbox=bbox,
            task_model=task_model,
            parser=self.parser,
            ocr_inference_pool=self._ocr_task_executor_pool,
            describe_picture_pool=self._describe_picture_task_executor_pool,
        )
        task_result, _ = await asyncio.create_task(self._concurrent_run(task))
        retry_run = self.parser.page_retry_number
        while task_result is None and retry_run > 0:
            logger.info(f"Retrying parsing image {job_response.input_s3_path}")
            task_result, _ = await asyncio.create_task(self._concurrent_run(task))
            retry_run = retry_run - 1

        if task_result is None:
            job_response.task_stats.failed_task_count = 1
            raise RuntimeError(
                f"Failed to parse image {job_response.input_s3_path} "
                f"after {self.parser.page_retry_number} retries"
            )

        if task.status == "fallback":
            job_response.task_stats.fallback_task_count = 1
        else:
            job_response.task_stats.finished_task_count = 1
        return task_result, task.token_usage

    async def _rebuild_directory(self, cells_list, images_origin):
        if self.directory_cleaner is None:
            self.directory_cleaner = DirectoryCleaner()

        await self.directory_cleaner.reset_header_level(cells_list, images_origin)

    async def parse_pdf(
        self,
        input_path,
        filename,
        prompt_mode,
        save_dir,
        rebuild_directory=False,
        describe_picture=False,
    ):
        loop = asyncio.get_running_loop()

        print(f"Loading PDF: {input_path}")
        # Run blocking PDF loading in executor
        images_origin, scale_factors = await loop.run_in_executor(
            self.parser.cpu_executor, load_images_from_pdf, input_path
        )

        total_pages = len(images_origin)
        print(
            f"Parsing PDF with {total_pages} pages using concurrency of {self.parser.concurrency_limit}..."
        )

        semaphore = asyncio.Semaphore(self.parser.concurrency_limit)

        async def worker(page_idx, image):
            async with semaphore:
                return await self.parser._parse_single_image(
                    origin_image=image,
                    prompt_mode=prompt_mode,
                    save_dir=(
                        None if rebuild_directory or describe_picture else save_dir
                    ),
                    save_name=(
                        None if rebuild_directory or describe_picture else filename
                    ),
                    source="pdf",
                    page_idx=page_idx,
                    scale_factor=scale_factors[page_idx],
                )

        tasks = [worker(i, image) for i, image in enumerate(images_origin)]

        if not describe_picture and not rebuild_directory:
            results = await tqdm.gather(*tasks, desc="Processing PDF pages")
            results.sort(key=lambda x: x["page_no"])
            return results

        cells_list = await tqdm.gather(*tasks, desc="Processing PDF pages")
        cells_list.sort(key=lambda x: x["page_no"])

        if describe_picture:

            async def worker_des(page_idx, image):
                async with semaphore:
                    return await self.parser._describe_picture_in_single_page(
                        origin_image=image,
                        cells=cells_list[page_idx],
                    )

            tasks = [worker_des(i, image) for i, image in enumerate(images_origin)]
            await tqdm.gather(*tasks, desc="extracting infomation from picture")

        if rebuild_directory:
            await self._rebuild_directory(cells_list, images_origin)

        results = []
        for cell in cells_list:
            save_name_page = f"{filename}_page_{cell['page_no']}"
            result = await self.parser.save_results(
                cell,
                save_dir,
                save_name_page,
                images_origin[cell["page_no"]],
                scale_factors[cell["page_no"]],
            )
            results.append(result)

        return results

    async def _concurrent_run(self, task):
        async with self.parser.semaphore:
            return await task.run(), task

    # TODO(zihao): rebuild_directory
    async def schedule_pdf_tasks(
        self,
        job_response: JobResponseModel,
    ):
        job_files = job_response.get_job_local_files()
        with fitz.open(str(job_files.input_file_path)) as doc:
            pdf_page_num = doc.page_count
            logger.info(
                f"Scheduling {pdf_page_num} PDF tasks for {job_files.input_file_path}"
            )
            tasks = []
            job_response.task_stats.total_task_count = pdf_page_num

            failed_tasks = []
            with tqdm(
                total=pdf_page_num, desc="Processing PDF pages (schedule)"
            ) as pbar:
                for page_index in range(pdf_page_num):
                    page_file_name = (
                        f"{job_response.output_file_name}_page_{page_index}"
                    )
                    task_model = OcrTaskModel(
                        job_response=job_response,
                        task_id=str(page_index),
                        output_file_name=page_file_name,
                    )
                    task = PdfOcrTask(
                        doc[page_index],
                        task_model=task_model,
                        parser=self.parser,
                        ocr_inference_pool=self._ocr_task_executor_pool,
                        describe_picture_pool=self._describe_picture_task_executor_pool,
                        storage_manager=self._storage_manager,
                    )
                    tasks.append(asyncio.create_task(self._concurrent_run(task)))

                # retry by stages instead of retrying each task asynchronously
                for future in asyncio.as_completed(tasks):
                    task_result, task = await future
                    if task_result is None:
                        failed_tasks.append(task)
                        if task.error_msg is not None:
                            logger.warning(
                                f"Error processing task {task.job_id}-{task.task_id}: "
                                f"{task.error_msg}"
                            )
                        continue
                    pbar.update(1)
                    yield task_result, task.status, task.token_usage

                retry_run = self.parser.page_retry_number
                while len(failed_tasks) > 0 and retry_run > 0:
                    task_ids = [task.task_id for task in failed_tasks]
                    logger.info(f"Retrying parsing pages {','.join(task_ids)}")
                    retry_run = retry_run - 1
                    tasks = []
                    for task in failed_tasks:
                        tasks.append(asyncio.create_task(self._concurrent_run(task)))
                    failed_tasks = []
                    for future in asyncio.as_completed(tasks):
                        task_result, task = await future
                        if task_result is None:
                            failed_tasks.append(task)
                            if task.error_msg is not None:
                                logger.warning(
                                    f"Error processing task {task.job_id}-{task.task_id}: "
                                    f"{task.error_msg}"
                                )
                            continue
                        pbar.update(1)
                        yield task_result, task.status, task.token_usage

                if len(failed_tasks) > 0:
                    task_ids = [task.task_id for task in failed_tasks]
                    error_msg = (
                        f"Failed to process {len(failed_tasks)} pages in "
                        f"{job_response.input_s3_path} after {self.parser.page_retry_number}"
                        f" retries: {task_ids}"
                    )
                    logger.error(error_msg)
                    for task in failed_tasks:
                        yield {
                            "page_no": int(task.task_id)
                        }, task.status, task.token_usage
