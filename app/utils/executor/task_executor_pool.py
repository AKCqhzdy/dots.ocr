import asyncio
from typing import List

from loguru import logger
from pydantic import BaseModel
from opentelemetry import trace
from app.utils.tracing import get_tracer


class TaskExecutorPool(BaseModel):
    max_queue_size: int = 4
    concurrent_task_limit: int = 4
    name: str = "TaskExecutorPool"

    _workers: List[asyncio.Task] = []
    # TODO(tatiana): use a priority queue to improve job completion time?
    _task_queue: asyncio.Queue = None

    def start(self):
        logger.info(f"Starting up {self.concurrent_task_limit} {self.name} workers...")
        self._task_queue = asyncio.Queue(self.max_queue_size)

        for i in range(self.concurrent_task_limit):
            task = asyncio.create_task(self._worker_loop(f"TaskWorker-{i}"))
            self._workers.append(task)

    def stop(self):
        for worker in self._workers:
            worker.cancel()
        asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("All worker tasks have been stopped.")

    async def add_task(self, task):
        await self._task_queue.put(task)

    async def _worker_loop(self, worker_id: str):
        logger.debug(f"{worker_id} started")
        while True:
            try:
                task = await self._task_queue.get()
                await task.process()
                self._task_queue.task_done()
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} is shutting down.")
                break


class BatchTaskExecutorPool():
    def __init__(
        self,
        batch_handler: callable,
        max_queue_size: int = 2,
        concurrent_task_limit: int = 2,
        worker_num: int = 1,
        batch_collect_window: float = 0.1,
        name: str = "BatchTaskExecutorPool",
    ):
        self.batch_handler = batch_handler
        self.max_queue_size = max_queue_size
        self.concurrent_task_limit = concurrent_task_limit
        self.worker_num = worker_num
        self.batch_collect_window = batch_collect_window
        self.name = name
        self._workers = []
        self._task_queue = None

    def start(self):
        logger.info(f"Starting up {self.worker_num} {self.name} workers...")
        self._task_queue = asyncio.Queue(self.max_queue_size)

        for i in range(self.worker_num):
            task = asyncio.create_task(self._worker_loop(f"TaskWorker-{i}"))
            self._workers.append(task)

    def stop(self):
        for worker in self._workers:
            worker.cancel()
        asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("All worker tasks have been stopped.")
        
    async def add_task(self, task):
        await self._task_queue.put(task)

    async def _worker_loop(self, worker_id: str):
        logger.debug(f"{worker_id} started")
        while True:
            batch = []
            try:
                loop = asyncio.get_running_loop()
                start_time = loop.time()
                while len(batch) < self.concurrent_task_limit:
                    timeout_left = self.batch_collect_window - (loop.time() - start_time)
                    if timeout_left <= 0:
                        break
                    try:
                        task = await asyncio.wait_for(self._task_queue.get(), timeout=timeout_left)
                        batch.append(task)
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} is shutting down.")
                break
            
            if not batch:
                continue
            
            links = []
            for task in batch:
                if task._span and task._span.is_recording():
                    links.append(trace.Link(task._span.get_span_context()))

            # TODO(zihao): find a better way to trace batch processing
            # with get_tracer().start_as_current_span(
            #     f"{self.name}.process_batch",
            #     links=links
            # ) as batch_span:
            try:
                logger.info(f"Processing layout detection batch of size {len(batch)}.")
                # batch_span.set_attribute("batch.size", len(batch))
                
                images_to_process = [task._image for task in batch]
                futures = [task._completion_future for task in batch]

                results = await self.batch_handler(images_to_process)

                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
            except asyncio.CancelledError:
                logger.warning(f"[{self.name}] Worker was cancelled. The current batch of size {len(batch)} might be lost.")
                # batch_span.set_status(trace.Status(trace.StatusCode.ERROR, "Worker cancelled"))
                for future in futures:
                    if not future.done():
                        future.set_exception(asyncio.CancelledError("The processing worker was cancelled."))
                break
            except Exception as e:
                logger.error(f"[{self.name}] Error processing batch: {e}", exc_info=True)
                # batch_span.record_exception(e)
                # batch_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                for future in futures:
                    if not future.done():
                        future.set_exception(e)