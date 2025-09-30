"""
Environment variables:
- Required:
  - POSTGRES_URL_NO_SSL_DEV: for establishing connection to the PostgreSQL database
- Optional:
  - OSS_ENDPOINT: the endpoint for accessing the OSS storage
  - OSS_ACCESS_KEY_ID: the access key for accessing the OSS storage
  - OSS_ACCESS_KEY_SECRET: the secret key for accessing the OSS storage

File resources:
- app/input: the directory for storing the input files. Created on startup if not exists.
- app/output: the directory for storing the output files. Created on startup if not exists.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from app.utils.hash import compute_md5_file, compute_md5_string
from app.utils.pg_vector import OCRTable, PGVector, status_type
from app.utils.storage import StorageManager
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import MAX_PIXELS, MIN_PIXELS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class JobResponseModel(BaseModel):
    job_id: str
    created_by: str = "system"
    updated_by: str = "system"
    created_at: datetime = None
    updated_at: datetime = None
    knowledgebase_id: str
    workspace_id: str
    # TODO(xxx): canceled is not supported yet
    status: status_type
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
            createdAt=self.created_at,
            updatedAt=self.updated_at,
        )


######################################## Constants ########################################

# Number of concurrent jobs that can run.
NUM_WORKERS = 4
RETRY_TIMES = 3
DPI = 200
# This is the vllm URL for health check
TARGET_URL = "http://localhost:8000/health"
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

######################################## Resources ########################################

global_lock_manager = asyncio.Lock()
pgvector_lock = asyncio.Lock()
processing_input_locks = {}
# FIXME(tatiana): if input is the same but output path differs, the computation is repeated
processing_output_locks = {}

dots_parser = DotsOCRParser(
    ip="localhost",
    port=8000,
    dpi=DPI,
    concurrency_limit=8,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)
storage_manager = StorageManager()
pg_vector_manager = PGVector()

# TODO(tatiana): failure recovery from persistent storage
job_response_dict: Dict[str, JobResponseModel] = {}
job_queue = asyncio.Queue()


@asynccontextmanager
async def lifespan(_: FastAPI):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    await pg_vector_manager.ensure_table_exists()

    logging.info("Starting up %s worker tasks...", NUM_WORKERS)
    worker_tasks: List[asyncio.Task] = []
    for i in range(NUM_WORKERS):
        task = asyncio.create_task(worker(f"Worker-{i}"))
        worker_tasks.append(task)

    yield

    logging.info("Shutting down and canceling worker tasks...")
    for task in worker_tasks:
        task.cancel()

    await asyncio.gather(*worker_tasks, return_exceptions=True)
    logging.info("All worker tasks have been canceled.")


app = FastAPI(
    title="dotsOCR API",
    description="API for PDF and image text recognition using dotsOCR by Grant",
    version="1.0.0",
    lifespan=lifespan,
)


def parse_s3_path(s3_path: str, is_s3):
    if is_s3:
        s3_path = s3_path.replace("s3://", "")
    else:
        s3_path = s3_path.replace("oss://", "")
    bucket, *key_parts = s3_path.split("/")
    return bucket, "/".join(key_parts)


async def update_pgvector(job: JobResponseModel):
    """Insert a new job or update an existing job in the PG database.

    Args:
        job (JobResponseModel): The job to be inserted or updated.
    """
    job.updated_at = datetime.now(UTC)
    # TODO(tatiana): Why do we need to use a lock here? Consider measuring the performance impact.
    async with pgvector_lock:
        # await pg_vector_manager.ensure_table_exists()
        record = job.get_table_record()
        await pg_vector_manager.upsert_record(record)

        # If there is a problem, it may not be reported immediately after the operation.
        # Use flush to force the operation to be committed, so that the error can be
        # reported immediately if exists.
        # await pg_vector_manager.flush()


async def get_record_pgvector(job_id: str) -> OCRTable:
    async with pgvector_lock:
        # await pg_vector_manager.ensure_table_exists()
        record = await pg_vector_manager.get_record_by_id(job_id)
        return record


@app.post("/status")
async def status_check(ocr_job_id: str = Form(alias="OCRJobId")):
    # if ocr_job_id not in JobResponseDict:
    #     raise HTTPException(status_code=404, detail=f"Job ID {ocr_job_id} not found")

    return await get_record_pgvector(ocr_job_id)


async def stream_and_upload_generator(job_response: JobResponseModel):
    input_s3_path = job_response.input_s3_path
    output_s3_path = job_response.output_s3_path
    is_s3 = job_response.is_s3

    try:

        file_bucket, file_key = parse_s3_path(input_s3_path, is_s3)
        input_file_path = INPUT_DIR / file_bucket / file_key
        input_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with global_lock_manager:
            if input_s3_path not in processing_input_locks:
                processing_input_locks[input_s3_path] = asyncio.Lock()
            if output_s3_path not in processing_output_locks:
                processing_output_locks[output_s3_path] = asyncio.Lock()
        input_lock = processing_input_locks[input_s3_path]
        output_lock = processing_output_locks[output_s3_path]

        async with input_lock:
            async with output_lock:

                # download file from S3
                try:
                    await storage_manager.download_file(
                        bucket=file_bucket,
                        key=file_key,
                        local_path=str(input_file_path),
                        is_s3=is_s3,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download file from s3/oss: {str(e)}"
                    ) from e

                # compute MD5 hash of the input file
                try:
                    file_md5 = (
                        job_response.job_id
                        + ":"
                        + compute_md5_file(str(input_file_path))
                    )
                    logging.info(
                        "MD5 hash of input file %s: %s", input_s3_path, file_md5
                    )
                except Exception as e:
                    logging.error(
                        "Failed to compute MD5 hash for %s: %s", input_s3_path, e
                    )
                    raise RuntimeError(f"Failed to compute MD5 hash: {str(e)}") from e

                # prepare local path
                output_bucket, output_key = parse_s3_path(output_s3_path, is_s3)
                output_file_name = output_s3_path.rstrip("/").rsplit("/", 1)[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(
                    output_md_path.stem + "_nohf"
                ).with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md5_path = output_md_path.with_suffix(".md5")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                output_file_path.mkdir(parents=True, exist_ok=True)

                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = (
                    await storage_manager.check_existing_results_sync(
                        bucket=output_bucket,
                        prefix=f"{output_key}/{output_file_name}",
                        is_s3=is_s3,
                    )
                )

                # If so, download md5 file and compare hashes
                if md5_exists:
                    try:
                        await storage_manager.download_file(
                            bucket=output_bucket,
                            key=f"{output_key}/{output_file_name}.md5",
                            local_path=str(output_md5_path),
                            is_s3=is_s3,
                        )
                        with open(output_md5_path, "r", encoding="utf-8") as f:
                            existing_md5 = f.read().strip()
                        if existing_md5 == file_md5 and not JobResponse.overwrite:
                            if all_files_exist:
                                logging.info(
                                    "Output files already exist in S3 and MD5 matches for %s. Skipping processing.",
                                    input_s3_path,
                                )
                                job_response.json_url = (
                                    f"{output_s3_path}/{output_file_name}.json"
                                )
                                job_response.md_url = (
                                    f"{output_s3_path}/{output_file_name}.md"
                                )
                                job_response.md_nohf_url = (
                                    f"{output_s3_path}/{output_file_name}_nohf.md"
                                )
                                job_response.page_prefix = (
                                    f"{output_s3_path}/{output_file_name}_page_"
                                )
                                skip_response = {
                                    "success": True,
                                    "total_pages": 0,
                                    "output_s3_path": output_s3_path,
                                    "message": "Output files already exist and MD5 matches. Skipped processing.",
                                }
                                yield json.dumps(skip_response) + "\n"
                                return
                            logging.info(
                                "MD5 matches for %s, but some output files are missing. Reprocessing the file.",
                                input_s3_path,
                            )
                        else:
                            # clean the whole output directory in S3
                            # print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                            # await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                            logging.info(
                                f"MD5 mismatch for {input_s3_path} or overwrite is set true. Reprocessing the file."
                            )
                    except Exception as e:
                        logging.warning(
                            "Failed to verify existing MD5 hash for %s: %s. Reprocessing the file.",
                            input_s3_path,
                            e,
                        )
                else:
                    # clean the whole output directory in S3 for safety
                    # print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                    # logging.info(f"No MD5 hash found for {input_s3_path}. Cleaning output directory.")
                    # await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    logging.info(
                        "No MD5 hash found for %s. Reprocessing the file.",
                        input_s3_path,
                    )

                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                with open(output_md5_path, "w", encoding="utf-8") as f:
                    f.write(file_md5)
                logging.info("Saved MD5 hash to %s", output_md5_path)

                # Upload MD5 hash file to S3/OSS
                try:
                    await storage_manager.upload_file(
                        output_bucket,
                        f"{output_key}/{output_file_name}.md5",
                        str(output_md5_path),
                        is_s3,
                    )
                except Exception as e:
                    logging.warning("Failed to upload MD5 hash file to s3/oss: %s", e)

                # print(output_bucket, output_key)
                # print(output_file_name)
                # print(output_file_path)
                # print(output_md_path)

                job_response.json_url = f"{output_s3_path}/{output_file_name}.json"
                job_response.md_url = f"{output_s3_path}/{output_file_name}.md"
                job_response.md_nohf_url = (
                    f"{output_s3_path}/{output_file_name}_nohf.md"
                )
                job_response.page_prefix = f"{output_s3_path}/{output_file_name}_page_"

                # parse the PDF file and upload each page's output files
                all_paths_to_upload = []

                try:
                    if JobResponse.parse_type == "image":
                        result = await dots_parser.parse_image(
                            input_path=str(input_file_path),
                            filename=output_file_name,
                            prompt_mode=JobResponse.prompt_mode,
                            save_dir=output_file_path,
                            fitz_preprocess=JobResponse.fitz_preprocess,
                            describe_picture=JobResponse.describe_picture,
                        )
                        all_paths_to_upload.append(result)
                    else:
                        async for result in dots_parser.parse_pdf_stream(
                            input_path=input_file_path,
                            filename=Path(input_file_path).stem,
                            prompt_mode=JobResponse.prompt_mode,
                            save_dir=output_file_path,
                            rebuild_directory=JobResponse.rebuild_directory,
                            describe_picture=JobResponse.describe_picture,
                        ):
                            page_no = result.get("page_no", -1)

                            page_upload_tasks = []
                            paths_to_upload = {
                                "md": result.get("md_content_path"),
                                "md_nohf": result.get("md_content_nohf_path"),
                                "json": result.get("layout_info_path"),
                            }
                            for file_type, local_path in paths_to_upload.items():
                                if local_path:
                                    file_name = Path(local_path).name
                                    s3_key = f"{output_key}/{file_name}"
                                    task = asyncio.create_task(
                                        storage_manager.upload_file(
                                            output_bucket, s3_key, local_path, is_s3
                                        )
                                    )
                                    page_upload_tasks.append(task)
                            uploaded_paths_for_page = await asyncio.gather(
                                *page_upload_tasks
                            )

                            paths_to_upload["page_no"] = page_no
                            all_paths_to_upload.append(paths_to_upload)
                            page_response = {
                                "success": True,
                                "message": "parse success",
                                "page_no": page_no,
                                "uploaded_files": [
                                    path for path in uploaded_paths_for_page if path
                                ],
                            }
                            yield json.dumps(page_response) + "\n"
                except Exception as e:
                    logging.error("Error during parsing pages: %s", e)

                if JobResponse.parse_type == "pdf":
                    # combine all page to upload
                    all_paths_to_upload.sort(key=lambda item: item["page_no"])
                    output_files = {}
                    try:
                        output_files["md"] = open(output_md_path, "w", encoding="utf-8")
                        output_files["json"] = open(
                            output_json_path, "w", encoding="utf-8"
                        )
                        output_files["md_nohf"] = open(
                            output_md_nohf_path, "w", encoding="utf-8"
                        )
                        all_json_data = []
                        for p in all_paths_to_upload:
                            page_no = p.pop("page_no")
                            for file_type, local_path in p.items():
                                if file_type == "json":
                                    try:
                                        with open(
                                            local_path, "r", encoding="utf-8"
                                        ) as input_file:
                                            data = json.load(input_file)
                                        data = {"page_no": page_no, **data}
                                        all_json_data.append(data)
                                    except Exception as e:
                                        print(
                                            f"WARNING: Failed to read layout info file {local_path}: {str(e)}"
                                        )
                                        all_json_data.append({"page_no": page_no})
                                else:
                                    try:
                                        with open(
                                            local_path, "r", encoding="utf-8"
                                        ) as input_file:
                                            file_content = input_file.read()
                                        output_files[file_type].write(file_content)
                                        output_files[file_type].write("\n\n")
                                    except Exception as e:
                                        print(
                                            f"WARNING: Failed to read {file_type} file {local_path}: {str(e)}"
                                        )
                        json.dump(
                            all_json_data,
                            output_files["json"],
                            indent=4,
                            ensure_ascii=False,
                        )
                    finally:
                        # Ensure all file handles are properly closed
                        for file_handle in output_files.values():
                            if hasattr(file_handle, "close"):
                                file_handle.close()

                await storage_manager.upload_file(
                    output_bucket,
                    f"{output_key}/{output_file_name}.md",
                    str(output_md_path),
                    is_s3,
                )
                await storage_manager.upload_file(
                    output_bucket,
                    f"{output_key}/{output_file_name}_nohf.md",
                    str(output_md_nohf_path),
                    is_s3,
                )
                await storage_manager.upload_file(
                    output_bucket,
                    f"{output_key}/{output_file_name}.json",
                    str(output_json_path),
                    is_s3,
                )

                final_response = {
                    "success": True,
                    "total_pages": len(all_paths_to_upload),
                    "output_s3_path": output_s3_path,
                }
                yield json.dumps(final_response) + "\n"

    except Exception as e:
        error_msg = json.dumps({"success": False, "detail": str(e)})
        yield error_msg + "\n"


@retry(
    stop=stop_after_attempt(RETRY_TIMES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)
async def attempt_to_process_job(job: JobResponseModel):
    attempt_num = attempt_to_process_job.retry.statistics.get("attempt_number", 0) + 1
    if attempt_num == 1:
        job.status = "processing"
        await update_pgvector(job)
    else:
        job.status = "retrying"
        await update_pgvector(job)
    job.message = f"Processing job, attempt number: {attempt_num}"

    try:
        async for page_result in stream_and_upload_generator(job):
            pass
    except Exception as e:
        logging.error(
            "Job %s failed on attempt %s with error: %s", job.job_id, attempt_num, e
        )
        raise


async def worker(worker_id: str):
    print(f"{worker_id} started")
    while True:
        try:
            job_id = await job_queue.get()

            job_response = job_response_dict.get(job_id)
            if not job_response:
                logging.error(
                    "%s: Job ID '%s' found in queue but not in JobResponseDict. Discarding task.",
                    worker_id,
                    job_id,
                )
                job_queue.task_done()
                continue

            try:
                await attempt_to_process_job(job_response)
                logging.info("Job %s successfully processed.", job_response.job_id)
                job_response.status = "completed"
                job_response.message = "Job completed successfully"
            except Exception as e:
                logging.error(
                    "Job %s failed after %s attempts. Final error: %s",
                    job_response.job_id,
                    RETRY_TIMES,
                    e,
                    exc_info=True,
                )
                job_response.status = "failed"
                job_response.message = (
                    f"Job failed after multiple retries. Final error: {str(e)}"
                )
            await update_pgvector(job_response)
            job_queue.task_done()

        except asyncio.CancelledError:
            logging.info("%s is shutting down.", worker_id)
            break
        except Exception as e:
            logging.error("Unexpected error in %s: %s", worker_id, e)


@app.post("/parse/file")
async def parse_file(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    knowledgebase_id: str = Form(alias="knowledgebaseId"),
    workspace_id: str = Form(alias="workspaceId"),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = Form(False),
    rebuild_directory: bool = Form(False),
    describe_picture: bool = Form(True),
    overwrite: bool = Form(False),
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except (TypeError, AttributeError) as err:
        raise HTTPException(
            status_code=400, detail="Invalid input_s3_path format"
        ) from err

    supported_formats = [".pdf", ".jpg", ".jpeg", ".png"]
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats are: {', '.join(supported_formats)}",
        )

    # Current logic: only if input, output, knowledgebaseId, workspaceId are all the same, we consider it as the same job, and add job_id to the md5 file
    ocr_job_id = "job-" + compute_md5_string(
        f"{input_s3_path}_{output_s3_path}_{knowledgebase_id}_{workspace_id}"
    )

    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")

    # Get the existing job status from pgvector
    existing_record = await get_record_pgvector(ocr_job_id)
    if existing_record and ocr_job_id in job_response_dict:
        if existing_record.status in ["pending", "retrying", "processing"]:
            return JSONResponse(
                {
                    "OCRJobId": ocr_job_id,
                    "status": existing_record.status,
                    "message": "Job is already in progress",
                },
                status_code=202,
            )
        if existing_record.status in ["completed", "failed", "canceled"]:
            # allow re-process but check md5 first in the worker
            pass

    job_response = JobResponseModel(
        job_id=ocr_job_id,
        created_at=datetime.now(UTC).replace(tzinfo=None),
        updated_at=datetime.now(UTC).replace(tzinfo=None),
        status="pending",
        knowledgebase_id=knowledgebase_id,
        workspace_id=workspace_id,
        message="Job is pending",
        is_s3=is_s3,
        input_s3_path=input_s3_path,
        output_s3_path=output_s3_path,
        parse_type="pdf" if file_ext == ".pdf" else "image",
        prompt_mode=prompt_mode,
        fitz_preprocess=fitz_preprocess,
        rebuild_directory=rebuild_directory,
        describe_picture=describe_picture,
        overwrite=overwrite,
    )
    logging.info("Job %s created. %s", ocr_job_id, job_response)
    job_response_dict[ocr_job_id] = job_response
    await update_pgvector(job_response)
    await job_queue.put(ocr_job_id)

    return JSONResponse({"OCRJobId": ocr_job_id}, status_code=200)


@app.post("/parse/image_old")
async def parse_image_old(**kwargs):
    raise HTTPException(
        status_code=400, detail="Deprecated API, please use /parse/file instead"
    )


@app.post("/parse/pdf_old")
async def parse_pdf_old(**kwargs):
    raise HTTPException(
        status_code=400, detail="Deprecated API, please use /parse/file instead"
    )


@app.post("/parse/file_old")
async def parse_file_old(**kwargs):
    raise HTTPException(
        status_code=400, detail="Deprecated API, please use /parse/file instead"
    )


async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(TARGET_URL, timeout=5.0)

        headers_to_exclude = {
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "connection",
        }
        proxied_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in headers_to_exclude
        }

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=proxied_headers,
            media_type=response.headers.get("content-type"),
        )
    except httpx.ConnectError as e:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": 503,
                "detail": f"Health check failed: Unable to connect to DotsOCR service. Error: {e}",
            },
        )
    except httpx.TimeoutException as e:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "status": 504,
                "detail": f"Health check failed: Request to DotsOCR service timed out. Error: {e}",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": 500,
                "detail": f"An unexpected error occurred during health check. Error: {e}",
            },
        )


@app.get("/health")
async def health():
    return await health_check()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6008, reload=True)
