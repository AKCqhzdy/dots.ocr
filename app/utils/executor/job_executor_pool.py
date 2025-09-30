import asyncio
from datetime import datetime
from typing import Awaitable, Callable, Dict, List

from app.utils.pg_vector import JobStatusType, OCRTable
from loguru import logger
from pydantic import BaseModel


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
    is_s3: bool = True
    parse_type: str = "pdf"

    input_s3_path: str
    output_s3_path: str
    page_prefix: str = None
    json_url: str = None
    md_url: str = None
    md_nohf_url: str = None

    prompt_mode: str = "prompt_layout_all_en"
    fitz_preprocess: bool = False
    rebuild_directory: bool = False
    describe_picture: bool = False
    overwrite: bool = False

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
    Not thread-safe.  Separate job states from job execution.
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
            logger.info("Job %s is cancelled.", self.job_response.job_id)
            await self._set_cancelled()
            return

        try:
            await self._execute(self.job_response)
            logger.success("Job %s successfully processed.", self.job_response.job_id)
            await self._set_finished()
        except Exception as e:
            logger.error(
                "Job %s failed. Final error: %s",
                self.job_response.job_id,
                e,
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


# TODO(tatiana): failure recovery. recover job from persistent storage
class JobExecutorPool(BaseModel):
    max_queue_size: int = 4
    pool_size: int = 4
    job_retry_times: int = 3

    _workers: List[asyncio.Task] = []
    _job_queue: asyncio.Queue = None
    _job_dict: Dict[str, Job] = {}

    def start(self):
        logger.info("Starting up %s workers...", self.pool_size)
        self._job_queue = asyncio.Queue(self.max_queue_size)

        for i in range(self.pool_size):
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
        logger.info(f"{worker_id} started")
        while True:
            try:
                job_id = await self._job_queue.get()
                job = self._job_dict.get(job_id)

                # This is unlikely since add_job should always save job_response
                # to self.job_response_dict
                if not job:
                    logger.error(
                        "%s: Job ID '%s' found in queue but not in JobResponseDict."
                        " Discarding task.",
                        worker_id,
                        job_id,
                    )
                    self._job_queue.task_done()
                    continue

                await job.process()
                self._job_queue.task_done()

            except asyncio.CancelledError:
                logger.info("Worker %s is shutting down.", worker_id)
                break
            except Exception as e:
                logger.error("Unexpected error in worker %s: %s", worker_id, e)
