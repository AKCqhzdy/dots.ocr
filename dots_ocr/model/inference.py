import asyncio
import json
import os
import time
from typing import List, Optional, Union

import httpx
from loguru import logger
from openai import APITimeoutError, AsyncOpenAI
from openai.types import CompletionUsage
from opentelemetry import trace
from PIL import Image
from pydantic import BaseModel

from app.utils.tracing import get_tracer, traced
from dots_ocr.utils.image_utils import PILimage_to_base64
from dots_ocr.utils.prompts import dict_promptmode_to_prompt


class InferenceTaskOptions(BaseModel):
    model_name: str
    model_host: str
    model_port: int
    temperature: float
    top_p: float
    max_completion_tokens: int
    timeout: Union[List[int], int]
    max_attempts: int = 3

    def get_timeout(self, attempt_run: int):
        attempt_index = attempt_run - 1
        if isinstance(self.timeout, list):
            if attempt_index >= len(self.timeout):
                return self.timeout[-1]
            return self.timeout[attempt_index]
        return self.timeout


class InferenceTaskStats(BaseModel):
    success_usage: Optional[CompletionUsage] = None
    is_fallback: bool = False
    attempt_num: int = 0


class InferenceTask:
    """Enviroment variable (required):
    - API_KEY: the API key for the OpenAI API.
    """

    def __init__(
        self,
        span: trace.Span,
        options: InferenceTaskOptions,
        task_id: str,
        image: Image.Image,
        prompt: str,
    ):
        self._span = span
        self._options = options
        self._stats = InferenceTaskStats()
        self._task_id = task_id
        self._image = image
        self._prompt = prompt
        self._client = None
        self._completion_future = asyncio.Future()
        self._last_failure_reason = []

    @property
    def model_address(self) -> str:
        return f"http://{self._options.model_host}:{self._options.model_port}/v1"

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def is_fallback(self) -> bool:
        return self._stats.is_fallback

    # Here we do not count input tokens for failures because now the model
    # is self-hosted and do not incur actual cost.
    @property
    def success_usage(self) -> tuple[str, Optional[CompletionUsage]]:
        return self._options.model_name, self._stats.success_usage

    # TODO(tatiana): use tenacity.retry?
    async def process(self):
        with trace.use_span(self._span, end_on_exit=True):
            while self._stats.attempt_num < self._options.max_attempts:
                self._span.add_event(f"attempt-{self._stats.attempt_num}")
                with get_tracer().start_as_current_span(
                    f"attempt-{self._stats.attempt_num}"
                ) as span:
                    self._stats.attempt_num += 1
                    try:
                        logger.debug(
                            f"Inference task {self.task_id} started (attempt {self._stats.attempt_num})"
                        )
                        start_time = time.perf_counter()
                        result = await self._run()
                        end_time = time.perf_counter()
                        elapsed = end_time - start_time
                        logger.trace(
                            f"Inference task {self.task_id} completed in {elapsed:.4f} seconds"
                        )
                        # logger.debug(
                        #     f"Inference task {self.task_id} completed in {elapsed:.4f} seconds"
                        #     f" with result: {result}"
                        # )
                        self._completion_future.set_result(result)
                        break
                    except APITimeoutError as e:  # retry on timeout
                        self._last_failure_reason.append(type(e).__name__)
                        if self.is_last_attempt():
                            self._completion_future.set_exception(e)
                        span.record_exception(e)
                    except Exception as e:
                        self._last_failure_reason.append(type(e).__name__)
                        logger.error(
                            f"Inference task {self.task_id} failed: {self._last_failure_reason[-1]} {e}",
                            exc_info=True,
                        )
                        self._completion_future.set_exception(e)
                        span.record_exception(e)
                        break

    def get_completion_future(self):
        return self._completion_future

    def is_last_attempt(self):
        return self._stats.attempt_num == self._options.max_attempts

    @traced()
    async def _run(self):
        return await self.inference_with_vllm()

    @traced()
    async def inference_with_vllm(self, prompt=None):
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=f'{os.environ.get("API_KEY", "0")}',
                base_url=self.model_address,
                timeout=6000,
                max_retries=0,
            )
        if prompt is None:
            prompt = self._prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": PILimage_to_base64(self._image)},
                    },
                    {
                        "type": "text",
                        "text": f"<|img|><|imgpad|><|endofimg|>{prompt}",
                    },  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
                ],
            }
        ]
        try:
            response = await self._client.chat.completions.create(
                messages=messages,
                model=self._options.model_name,
                max_completion_tokens=self._options.max_completion_tokens,
                temperature=self._options.temperature,
                top_p=self._options.top_p,
                timeout=self._options.get_timeout(self._stats.attempt_num),
            )
            self._stats.success_usage = response.usage
            response = response.choices[0].message.content
            return response
        except httpx.TimeoutException:
            logger.error(f"request timeout for task {self.task_id}")
            # TODO(tatiana): why except the error and return error str?
            # There is no handling logic for this result.
            return "timeout"
        except httpx.RequestError as e:
            logger.error(f"request error for task {self.task_id}: {e}")
            # TODO(tatiana): why except the error and return error str?
            # There is no handling logic for this result.
            return "error"

    # TODO(tatiana): make use of size to determine the queue
    # usage for more effective back pressure and memory management.
    def size(self):
        """The size of the image in bytes."""
        return len(self._image.tobytes())

    def token(self):
        """The number of token usage for processing the task."""


class OcrInferenceTask(InferenceTask):
    async def _run(self):
        if (
            len(self._last_failure_reason) > 0
            and self._last_failure_reason[-1] == "APITimeoutError"
        ):
            # Use fallback approach if this is the last attempt
            if self.is_last_attempt():
                # If there are consequtive timeout, use fallback approach
                result = await self._fallback_ocr()
                return result

        return await self.inference_with_vllm()

    async def _fallback_ocr(self):
        self._stats.is_fallback = True
        prompt = dict_promptmode_to_prompt["prompt_ocr"]
        # This can still timeout. Consider a better fallback approach?
        result = await self.inference_with_vllm(prompt)
        return json.dumps(
            [
                {
                    "bbox": [0, 0, self._image.width, self._image.height],
                    "category": "Text",
                    "text": result,
                }
            ]
        )

    async def _fallback_picture(self):
        # We may experience timeout during picture description if we have experienced
        # timeout during OCR inference. Using this fallback may not improve stability.
        self._stats.is_fallback = True
        logger.debug("Fall back to return picture...................")
        return json.dumps(
            [
                {
                    "bbox": [0, 0, self._image.width, self._image.height],
                    "category": "Picture",
                }
            ]
        )
