import asyncio
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Literal

from loguru import logger
from pydantic import BaseModel

from app.utils.pg_vector import JobStatusType, OCRTable
from app.utils.storage import parse_s3_path
from app.utils import configs


class JobLocalFiles(BaseModel):
    remote_input_bucket: str
    remote_input_file_key: str

    remote_output_bucket: str
    remote_output_file_key: str

    output_file_name: str

    @property
    def input_file_path(self):
        return configs.INPUT_DIR / self.remote_input_bucket / self.remote_input_file_key

    @property
    def output_dir_path(self):
        return (
            configs.OUTPUT_DIR / self.remote_output_bucket / self.remote_output_file_key
        )

    @property
    def output_json_path(self):
        output_file_path = self.output_dir_path / self.output_file_name
        return output_file_path.with_suffix(".json")

    @property
    def output_md_path(self):
        output_file_path = self.output_dir_path / self.output_file_name
        return output_file_path.with_suffix(".md")

    @property
    def output_md_nohf_path(self):
        return self.output_dir_path / f"{self.output_file_name}_nohf.md"

    @property
    def output_md5_path(self):
        output_file_path = self.output_dir_path / self.output_file_name
        return output_file_path.with_suffix(".md5")


class JobResponseModel(BaseModel):
    job_id: str
    created_by: str = "system"
    updated_by: str = "system"
    created_at: datetime = None
    updated_at: datetime = None
    knowledgebase_id: str
    workspace_id: str
    status: JobStatusType
    message: str
    parse_type: Literal["pdf", "image"] = "pdf"

    is_s3: bool = True
    input_s3_path: str
    output_s3_path: str

    prompt_mode: str = "prompt_layout_all_en"
    fitz_preprocess: bool = False
    rebuild_directory: bool = False
    describe_picture: bool = False
    overwrite: bool = False

    _job_local_files: JobLocalFiles = None

    def get_job_local_files(self):
        if self._job_local_files is None:
            input_bucket, input_file_key = parse_s3_path(self.input_s3_path, self.is_s3)
            output_bucket, output_file_key = parse_s3_path(
                self.output_s3_path, self.is_s3
            )
            output_file_name = self.output_s3_path.rstrip("/").rsplit("/", 1)[-1]
            self._job_local_files = JobLocalFiles(
                remote_input_bucket=input_bucket,
                remote_input_file_key=input_file_key,
                remote_output_bucket=output_bucket,
                remote_output_file_key=output_file_key,
                output_file_name=output_file_name,
            )
        return self._job_local_files

    @property
    def output_file_name(self):
        return self.get_job_local_files().output_file_name

    @property
    def json_url(self):
        return f"{self.output_s3_path}/{self.output_file_name}.json"

    @property
    def md_url(self):
        return f"{self.output_s3_path}/{self.output_file_name}.md"

    @property
    def md_nohf_url(self):
        return f"{self.output_s3_path}/{self.output_file_name}_nohf.md"

    @property
    def page_prefix(self):
        return f"{self.output_s3_path}/{self.output_file_name}_page_"

    def transform_to_map(self):
        mapping = {
            "url": self.output_s3_path,
            "knowledgebaseId": self.knowledgebase_id,
            "workspaceId": self.workspace_id,
            "markdownUrl": self.md_url,
            "jsonUrl": self.json_url,
            "status": self.status,
        }
        return {k: (v if v is not None else "") for k, v in mapping.items()}

    def get_table_record(self) -> OCRTable:
        return OCRTable(
            id=self.job_id,
            url=self.input_s3_path,
            markdownUrl=self.md_url,
            jsonUrl=self.json_url,
            status=self.status,
            createdBy=self.created_by,
            updatedBy=self.updated_by,
            createdAt=self.created_at.replace(tzinfo=None),
            updatedAt=self.updated_at.replace(tzinfo=None),
        )


class Job:
    """
    Not thread-safe. Separate job states from job execution.
    - Job states are stored in JobResponseModel and requires persistence.
    - Job execution logic is implemented in this class.
    """

    def __init__(
        self,
        job_response: JobResponseModel,
        execute: Callable[[JobResponseModel], Awaitable[None]],
        on_status_change: Callable[[JobResponseModel], Awaitable[None]],
    ):
        self.job_response = job_response
        self._on_status_change = on_status_change
        self._execute = execute
        self._cancel_requested = False

    async def process(self):
        # TODO(tatiana): handle the cancellation logic here. Now just do a trivial cancellation.
        if self._cancel_requested:
            logger.info(f"Job {self.job_response.job_id} is cancelled.")
            await self._set_cancelled()
            return

        try:
            await self._execute(self.job_response)
            logger.success(f"Job {self.job_response.job_id} successfully processed.")
            await self._set_finished()
        except Exception as e:
            logger.error(
                f"Job {self.job_response.job_id} failed. Final error: {e}",
                exc_info=True,
            )
            await self._set_failed(e)

    def cancel(self):
        # TODO(tatiana): handle the cancellation logic here. Now just do a trivial cancellation.
        self._cancel_requested = True

    async def restore(self):
        # TODO(tatiana): support failure recovery and resume processing
        #                in the middle of job execution.
        raise NotImplementedError(
            "Failure recovery is not supported yet. "
            f"Job {self.job_response.job_id} cannot be restored."
        )

    async def _set_failed(self, error):
        self.job_response.status = "failed"
        self.job_response.message = (
            f"Job failed after multiple retries. Final error: {str(error)}"
        )
        await self._on_status_change(self.job_response)

    async def _set_finished(self):
        self.job_response.status = "completed"
        self.job_response.message = "Job completed successfully"
        await self._on_status_change(self.job_response)

    async def _set_cancelled(self):
        self.job_response.status = "canceled"
        self.job_response.message = "Job is cancelled"
        await self._on_status_change(self.job_response)


# TODO(tatiana): Make jobs run in parallel?
# TODO(tatiana): failure recovery. recover job from persistent storage
class JobExecutorPool(BaseModel):
    max_queue_size: int = 4
    concurrent_job_limit: int = 4

    _workers: List[asyncio.Task] = []
    _job_queue: asyncio.Queue = None
    _job_dict: Dict[str, Job] = {}

    def start(self):
        logger.info(f"Starting up {self.concurrent_job_limit} workers...")
        self._job_queue = asyncio.Queue(self.max_queue_size)

        for i in range(self.concurrent_job_limit):
            task = asyncio.create_task(self._worker_loop(f"Worker-{i}"))
            self._workers.append(task)

    def stop(self):
        logger.info("Shutting down and canceling worker tasks...")
        for worker in self._workers:
            worker.cancel()
        asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("All worker tasks have been canceled.")

    def is_job_waiting(self, job_id: str) -> bool:
        return job_id in self._job_dict

    async def add_job(self, job: Job):
        """Add a job for async execution."""
        self._job_dict[job.job_response.job_id] = job
        await self._job_queue.put(job.job_response.job_id)

    async def cancel_job(self, job_id: str):
        """Cancel a job.
        If the job is already terminated or does not exist, there will be no effect.
        If the job is currently running, it will be asynchronously cancelled.
        """
        job = self._job_dict.get(job_id)
        if job:
            job.cancel()

    async def _worker_loop(self, worker_id: str):
        logger.debug(f"{worker_id} started")
        while True:
            try:
                job_id = await self._job_queue.get()
                job = self._job_dict.get(job_id)

                # This is unlikely since add_job should always save job_response
                # to self.job_response_dict
                if not job:
                    logger.error(
                        f"Worker {worker_id}: Job ID '{job_id}' found in queue "
                        "but not in JobResponseDict. Discarding task."
                    )
                    self._job_queue.task_done()
                    continue

                await job.process()
                self._job_queue.task_done()
                self._job_dict.pop(job_id)

            except asyncio.CancelledError:
                logger.info(f"{worker_id} is shutting down.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in {worker_id}: {e}")
