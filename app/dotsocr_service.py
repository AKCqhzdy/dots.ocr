from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Literal, List
from collections import deque
import os
from pathlib import Path
import tempfile
import uuid
import json
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from contextlib import asynccontextmanager 

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
import uvicorn
import logging
import asyncio
import httpx
import re
from app.utils.storage import StorageManager
from app.utils.hash import compute_md5_file, compute_md5_string
from app.utils.pg_vector import PGVector, OCRTable
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


NUM_WORKERS = 4 
WORKER_TASKS: List[asyncio.Task] = []
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Starting up {NUM_WORKERS} worker tasks...")
    for i in range(NUM_WORKERS):
        task = asyncio.create_task(worker(f"Worker-{i}"))
        WORKER_TASKS.append(task)
    
    yield

    logging.info("Shutting down and canceling worker tasks...")
    for task in WORKER_TASKS:
        task.cancel()
    
    await asyncio.gather(*WORKER_TASKS, return_exceptions=True)
    logging.info("All worker tasks have been canceled.")

app = FastAPI(
    title="dotsOCR API",
    description="API for PDF and image text recognition using dotsOCR by Grant",
    version="1.0.0",
    lifespan=lifespan 
)



BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

GLOBAL_LOCK_MANAGER = asyncio.Lock() 
PGVECTOR_LOCK = asyncio.Lock()
PROCESSING_INPUT_LOCKS = {} 
PROCESSING_OUTPUT_LOCKS = {} 

dots_parser = DotsOCRParser(
    ip="localhost",
    port=8000,
    dpi=200,
    concurrency_limit=8,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)


storage_manager = StorageManager()
pg_vector_manager = PGVector()

def parse_s3_path(s3_path: str, is_s3):
    if is_s3:
        s3_path = s3_path.replace("s3://", "")
    else:
        s3_path = s3_path.replace("oss://", "")
    bucket, *key_parts = s3_path.split("/")
    return bucket, "/".join(key_parts)

class JobResponseModel(BaseModel):
    job_id: str
    created_by: str = "system"
    updated_by: str = "system"
    created_at: datetime = None
    updated_at: datetime = None
    knowledgebase_id: str
    workspace_id: str
    status: Literal["pending", "retrying", "processing", "completed", "failed", "canceled"] # canceled havn't implemented
    message: str
    is_s3: bool = True

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

    def transform_to_map(self):
        mapping = {
            "url": self.output_s3_path,
            "knowledgebaseId": self.knowledgebase_id,
            "workspaceId": self.workspace_id,
            "markdownUrl": self.md_url,
            "jsonUrl": self.json_url,
            "status": self.status
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
            updatedAt=self.updated_at
        )

async def update_pgvector(job: JobResponseModel):
    job.updated_at = datetime.utcnow()
    async with PGVECTOR_LOCK:
        await pg_vector_manager.ensure_table_exists()
        
        record = await pg_vector_manager.get_record_by_id(job.job_id)

        if record:
            updates = job.get_table_record()
            await pg_vector_manager.update_record(job.job_id, updates)
        else:
            new_record = job.get_table_record()
            await pg_vector_manager.upsert_record(new_record)

        # await pg_vector_manager.flush()

async def get_record_pgvector(job_id: str) -> OCRTable:
    async with PGVECTOR_LOCK:
        await pg_vector_manager.ensure_table_exists()
        record = await pg_vector_manager.get_record_by_id(job_id)
        return record

JobResponseDict: Dict[str, JobResponseModel] = {}
JobLocks: Dict[str, asyncio.Lock] = {}
JobQueue = asyncio.Queue()

@app.post("/status")
async def status_check(OCRJobId: str = Form(...)):
    if OCRJobId not in JobResponseDict:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    return await get_record_pgvector(OCRJobId)

async def stream_and_upload_generator(
    JobResponse: JobResponseModel
):
    input_s3_path = JobResponse.input_s3_path
    output_s3_path = JobResponse.output_s3_path
    is_s3 = JobResponse.is_s3

    try:

        file_bucket, file_key = parse_s3_path(input_s3_path, is_s3)
        input_file_path = INPUT_DIR / file_bucket / file_key
        input_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with GLOBAL_LOCK_MANAGER:
            if input_s3_path not in PROCESSING_INPUT_LOCKS:
                PROCESSING_INPUT_LOCKS[input_s3_path] = asyncio.Lock()
            if output_s3_path not in PROCESSING_OUTPUT_LOCKS:
                PROCESSING_OUTPUT_LOCKS[output_s3_path] = asyncio.Lock()
        input_lock = PROCESSING_INPUT_LOCKS[input_s3_path]
        output_lock = PROCESSING_OUTPUT_LOCKS[output_s3_path]

        async with input_lock:
            async with output_lock:

                # download file from S3
                try:
                    await storage_manager.download_file(
                        bucket=file_bucket, key=file_key, local_path=str(input_file_path), is_s3 = is_s3
                    )
                    logging.info(f"download from s3/oss successfully: {input_s3_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from s3/oss: {str(e)}") from e
                
                # compute MD5 hash of the input file
                try:
                    file_md5 = JobResponse.job_id + ":" + compute_md5_file(str(input_file_path))
                    logging.info(f"MD5 hash of input file {input_s3_path}: {file_md5}")
                except Exception as e:
                    logging.error(f"Failed to compute MD5 hash for {input_s3_path}: {str(e)}")
                    raise RuntimeError(f"Failed to compute MD5 hash: {str(e)}") from e
                
                # prepare local path
                output_bucket, output_key = parse_s3_path(output_s3_path, is_s3)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md5_path = output_md_path.with_suffix(".md5")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                output_file_path.mkdir(parents=True, exist_ok=True)
            
                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = await storage_manager.check_existing_results_sync(
                    bucket=output_bucket, prefix=f"{output_key}/{output_file_name}", is_s3=is_s3
                )
            
                # If so, download md5 file and compare hashes
                if md5_exists:
                    try:
                        await storage_manager.download_file(
                            bucket=output_bucket,
                            key=f"{output_key}/{output_file_name}.md5",
                            local_path=str(output_md5_path),
                            is_s3=is_s3
                        )
                        with open(output_md5_path, 'r') as f:
                            existing_md5 = f.read().strip()
                        if existing_md5 == file_md5:
                            if all_files_exist:
                                logging.info(f"Output files already exist in S3 and MD5 matches for {input_s3_path}. Skipping processing.")
                                JobResponse.json_url = f"{output_s3_path}/{output_file_name}.json"
                                JobResponse.md_url = f"{output_s3_path}/{output_file_name}.md"
                                JobResponse.md_nohf_url = f"{output_s3_path}/{output_file_name}_nohf.md"
                                JobResponse.page_prefix = f"{output_s3_path}/{output_file_name}_page_"
                                skip_response = {
                                    "success": True,
                                    "total_pages": 0,
                                    "output_s3_path": output_s3_path,
                                    "message": "Output files already exist and MD5 matches. Skipped processing."
                                }
                                yield json.dumps(skip_response) + "\n"
                                return
                            logging.info(f"MD5 matches for {input_s3_path}, but some output files are missing. Reprocessing the file.")
                        else:
                            # clean the whole output directory in S3
                            # print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                            # await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                            logging.info(f"MD5 mismatch for {input_s3_path}. Reprocessing the file.")
                    except Exception as e:
                        logging.warning(f"Failed to verify existing MD5 hash for {input_s3_path}: {str(e)}. Reprocessing the file.")
                else:
                    # clean the whole output directory in S3 for safety
                    # print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                    # logging.info(f"No MD5 hash found for {input_s3_path}. Cleaning output directory.")
                    # await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    logging.info(f"No MD5 hash found for {input_s3_path}. Reprocessing the file.")

                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                with open(output_md5_path, 'w') as f:
                    f.write(file_md5)
                logging.info(f"Saved MD5 hash to {output_md5_path}")
                
                # Upload MD5 hash file to S3/OSS
                try:
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md5", str(output_md5_path), is_s3
                    )
                except Exception as e:
                    logging.warning(f"Failed to upload MD5 hash file to s3/oss: {str(e)}")

            
                # print(output_bucket, output_key)
                # print(output_file_name)
                # print(output_file_path)
                # print(output_md_path)

                JobResponse.json_url = f"{output_s3_path}/{output_file_name}.json"
                JobResponse.md_url = f"{output_s3_path}/{output_file_name}.md"
                JobResponse.md_nohf_url = f"{output_s3_path}/{output_file_name}_nohf.md"
                JobResponse.page_prefix = f"{output_s3_path}/{output_file_name}_page_"
                
                s3_prefix = f"{output_key}/{output_file_name}_page_"
 
                # parse the PDF file and upload each page's output files
                all_paths_to_upload = []

                try:
                    async for result in dots_parser.parse_pdf_stream(
                        input_path=input_file_path,
                        filename=Path(input_file_path).stem,
                        prompt_mode=JobResponse.prompt_mode,
                        save_dir=output_file_path,
                        rebuild_directory=JobResponse.rebuild_directory,
                        describe_picture=JobResponse.describe_picture
                    ):
                        page_no = result.get('page_no', -1)
                        
                        page_upload_tasks = []
                        paths_to_upload = {
                            'md': result.get('md_content_path'),
                            'md_nohf': result.get('md_content_nohf_path'),
                            'json': result.get('layout_info_path')
                        }
                        for file_type, local_path in paths_to_upload.items():
                            if local_path:
                                file_name = Path(local_path).name
                                s3_key = f"{output_key}/{file_name}"
                                task = asyncio.create_task(
                                    storage_manager.upload_file(output_bucket, s3_key, local_path, is_s3)
                                )
                                page_upload_tasks.append(task)
                        uploaded_paths_for_page = await asyncio.gather(*page_upload_tasks)                    

                        paths_to_upload['page_no'] = page_no
                        all_paths_to_upload.append(paths_to_upload)
                        page_response = {
                            "success": True,
                            "message": "parse success",
                            "page_no": page_no,
                            "uploaded_files": [path for path in uploaded_paths_for_page if path]
                        }
                        yield json.dumps(page_response) + "\n"
                except Exception as e:
                    logging.error(f"Error during parsing pages: {str(e)}")


                # combine all page to upload
                all_paths_to_upload.sort(key=lambda item: item['page_no'])
                output_files = {}
                try:
                    output_files['md'] = open(output_md_path, 'w', encoding='utf-8')
                    output_files['json'] = open(output_json_path, 'w', encoding='utf-8')
                    output_files['md_nohf'] = open(output_md_nohf_path, 'w', encoding='utf-8')
                    all_json_data = []
                    for p in all_paths_to_upload:
                        page_no = p.pop('page_no')
                        for file_type, local_path in p.items():
                            if file_type == 'json':
                                try:
                                    with open(local_path, 'r', encoding='utf-8') as input_file:
                                        data = json.load(input_file)
                                    data = {"page_no": page_no, **data}
                                    all_json_data.append(data)
                                except Exception as e:
                                    print(f"WARNING: Failed to read layout info file {local_path}: {str(e)}")
                                    all_json_data.append({"page_no": page_no})
                            else:
                                try:
                                    with open(local_path, 'r', encoding='utf-8') as input_file:
                                        file_content = input_file.read()
                                    output_files[file_type].write(file_content)
                                    output_files[file_type].write("\n\n")
                                except Exception as e:
                                    print(f"WARNING: Failed to read {file_type} file {local_path}: {str(e)}")
                    json.dump(all_json_data, output_files['json'], indent=4, ensure_ascii=False)
                finally:
                    # Ensure all file handles are properly closed
                    for file_handle in output_files.values():
                        if hasattr(file_handle, 'close'):
                            file_handle.close()
                
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}.md", str(output_md_path), is_s3)
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}_nohf.md", str(output_md_nohf_path), is_s3)
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}.json", str(output_json_path), is_s3)

                final_response = {
                    "success": True,
                    "total_pages": len(all_paths_to_upload),
                    "output_s3_path": output_s3_path
                }
                yield json.dumps(final_response) + '\n'
                        
    except Exception as e:
        error_msg = json.dumps({"success": False, "detail": str(e)})
        yield error_msg + "\n"


RETRY_TIMES = 3
@retry(
    stop=stop_after_attempt(RETRY_TIMES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True
)
async def attempt_to_process_job(job: JobResponseModel):
    attempt_num = attempt_to_process_job.retry.statistics.get('attempt_number', 0) + 1
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
        logging.error(f"Job {job.job_id} failed on attempt {attempt_num} with error: {e}")
        raise 

async def worker(worker_id: str):
    print(f"{worker_id} started")
    while True:
        try:
            job_id = await JobQueue.get()
            
            JobResponse = JobResponseDict.get(job_id)
            if not JobResponse:
                logging.error(f"{worker_id}: Job ID '{job_id}' found in queue but not in JobResponseDict. Discarding task.")
                JobQueue.task_done()
                continue

            try:
                await attempt_to_process_job(JobResponse)
                logging.info(f"Job {JobResponse.job_id} successfully processed.")
            except Exception as e:
                logging.error(f"Job {JobResponse.job_id} failed after 5 attempts. Final error: {e}", exc_info=True)
                JobResponse.status = "failed"
                JobResponse.message = f"Job failed after multiple retries. Final error: {str(e)}"
                await update_pgvector(JobResponse)

            JobResponse.status = "completed"
            JobResponse.message = "Job completed successfully"
            await update_pgvector(JobResponse)
            JobQueue.task_done()
            
        except asyncio.CancelledError:
            logging.info(f"{worker_id} is shutting down.")
            break
        except Exception as e:
            logging.error(f"Unexpected error in {worker_id}: {e}")

@app.post("/parse/file")
async def parse_file(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    knowledgebaseId: str = Form(...),
    workspaceId: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = Form(False),
    rebuild_directory: bool = Form(False),
    describe_picture: bool = Form(True)
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except (TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid input_s3_path format")

    supported_formats = ['.pdf', '.jpg', '.jpeg', '.png']
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats are: {', '.join(supported_formats)}"
        )
    
    # Current logic: only if input, output, knowledgebaseId, workspaceId are all the same, we consider it as the same job, and add job_id to the md5 file
    OCRJobId = "job-" + compute_md5_string(f"{input_s3_path}_{output_s3_path}_{knowledgebaseId}_{workspaceId}")
    
    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")

    # Get the existing job status from pgvector
    existing_record = await get_record_pgvector(OCRJobId)
    if existing_record:
        if existing_record.status in ["pending", "retrying", "processing"]:
            return JSONResponse({"OCRJobId": OCRJobId, "status": existing_record.status, "message": "Job is already in progress"}, status_code=202)
        elif existing_record.status == "completed" or existing_record.status == "failed":
            # allow re-process but check md5 first in the worker
            pass

    JobResponse = JobResponseModel(
        job_id=OCRJobId,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        status="pending",
        knowledgebase_id=knowledgebaseId,
        workspace_id=workspaceId,
        message="Job is pending",
        is_s3=is_s3,
        input_s3_path=input_s3_path,
        output_s3_path=output_s3_path,
        prompt_mode=prompt_mode,
        fitz_preprocess=fitz_preprocess,
        rebuild_directory=rebuild_directory,
        describe_picture=describe_picture
    )
    JobResponseDict[OCRJobId] = JobResponse
    JobLocks[OCRJobId] = asyncio.Lock()
    await update_pgvector(JobResponse)
    await JobQueue.put(OCRJobId)

    return JSONResponse({"OCRJobId": OCRJobId}, status_code=200)

#---------------------------not streamming---------------------------

async def parse(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False,
    parse_type: str = "pdf",  # or "image", default is "pdf"
    rebuild_directory: bool = False,
    describe_picture: bool = False
):
    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")
    try:
        file_bucket, file_key = parse_s3_path(input_s3_path, is_s3)
        input_file_path = INPUT_DIR / file_bucket / file_key
        input_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with GLOBAL_LOCK_MANAGER:
            if input_s3_path not in PROCESSING_INPUT_LOCKS:
                PROCESSING_INPUT_LOCKS[input_s3_path] = asyncio.Lock()
            if output_s3_path not in PROCESSING_OUTPUT_LOCKS:
                PROCESSING_OUTPUT_LOCKS[output_s3_path] = asyncio.Lock()
        input_lock = PROCESSING_INPUT_LOCKS[input_s3_path]
        output_lock = PROCESSING_OUTPUT_LOCKS[output_s3_path]
        
        async with input_lock:
            async with output_lock:
                # download file from S3
                try:
                    await storage_manager.download_file(
                        bucket=file_bucket, key=file_key, local_path=str(input_file_path), is_s3 = is_s3
                    )
                    logging.info(f"download from s3/oss successfully: {input_s3_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from s3/oss: {str(e)}") from e
                
                # compute MD5 hash of the input file
                try:
                    file_md5 = compute_md5_file(str(input_file_path))
                    logging.info(f"MD5 hash of input file {input_s3_path}: {file_md5}")
                except Exception as e:
                    logging.error(f"Failed to compute MD5 hash for {input_s3_path}: {str(e)}")
                    raise RuntimeError(f"Failed to compute MD5 hash: {str(e)}") from e

                output_bucket, output_key = parse_s3_path(output_s3_path, is_s3)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md5_path = output_md_path.with_suffix(".md5")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = await storage_manager.check_existing_results_sync(
                    bucket=output_bucket, prefix=f"{output_key}/{output_file_name}", is_s3=is_s3
                )
                
                # If so, download md5 file and compare hashes
                if md5_exists:
                    try:
                        await storage_manager.download_file(
                            bucket=output_bucket,
                            key=f"{output_key}/{output_file_name}.md5",
                            local_path=str(output_md5_path),
                            is_s3=is_s3
                        )
                        with open(output_md5_path, 'r') as f:
                            existing_md5 = f.read().strip()
                        if existing_md5 == file_md5:
                            if all_files_exist:
                                logging.info(f"Output files already exist in S3 and MD5 matches for {input_s3_path}. Skipping processing.")
                                return {
                                    "success": True,
                                    "total_pages": 0,
                                    "output_s3_path": output_s3_path,
                                    "message": "Output files already exist and MD5 matches. Skipped processing."
                                }
                            logging.info(f"MD5 matches for {input_s3_path}, but some output files are missing. Reprocessing the file.")
                        else:
                            # clean the whole output directory in S3
                            print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                            logging.info(f"MD5 mismatch for {input_s3_path}. Reprocessing the file.")
                            await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    except Exception as e:
                        logging.warning(f"Failed to verify existing MD5 hash for {input_s3_path}: {str(e)}. Reprocessing the file.")
                else:
                    # clean the whole output directory in S3 for safety
                    print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                    logging.info(f"No MD5 hash found for {input_s3_path}. Cleaning output directory.")
                    await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)

                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                with open(output_md5_path, 'w') as f:
                    f.write(file_md5)
                logging.info(f"Saved MD5 hash to {output_md5_path}")
                
                # Upload MD5 hash file to S3/OSS
                try:
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md5", str(output_md5_path), is_s3
                    )
                except Exception as e:
                    logging.warning(f"Failed to upload MD5 hash file to s3/oss: {str(e)}")

                # print(output_file_path)
                # print(output_file_name)
                # print(output_md_path)
                # try:
                if parse_type == "image":
                    results = await dots_parser.parse_image(
                        input_path=str(input_file_path),
                        filename=output_file_name,
                        prompt_mode=prompt_mode,
                        save_dir=output_file_path,
                        fitz_preprocess=fitz_preprocess
                    )
                else:
                    results = await dots_parser.parse_pdf(
                        input_path=input_file_path,
                        filename=output_file_name,
                        prompt_mode=prompt_mode,
                        save_dir=output_file_path,
                        rebuild_directory=rebuild_directory,
                        describe_picture=describe_picture,
                    )


                # Format results for all pages
                formatted_results = []
                all_md_content = []
                all_md_nohf_content = []
                for result in results:
                    layout_info_path = result.get('layout_info_path')
                    full_layout_info = {}
                    if layout_info_path and os.path.exists(layout_info_path):
                        try:
                            with open(layout_info_path, 'r', encoding='utf-8') as f:
                                full_layout_info = json.load(f)
                        except Exception as e:
                            print(f"WARNING: Failed to read layout info file: {str(e)}")
                    full_layout_info = {"page_no": result.get('page_no', -1), **full_layout_info}
                    formatted_results.append(full_layout_info)

                    md_content_path = result.get('md_content_path')
                    md_content = ""
                    if md_content_path and os.path.exists(md_content_path):
                        try:
                            with open(md_content_path, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                        except Exception as e:
                            print(f"WARNING: Failed to read markdown file: {str(e)}")
                    all_md_content.append(md_content)

                    md_content_nohf_path = result.get('md_content_nohf_path')
                    md_content_nohf = ""
                    if md_content_nohf_path and os.path.exists(md_content_nohf_path):
                        try:
                            with open(md_content_nohf_path, 'r', encoding='utf-8') as f:
                                md_content_nohf = f.read()
                        except Exception as e:
                            print(f"WARNING: Failed to read markdown file: {str(e)}")
                    all_md_nohf_content.append(md_content_nohf)

                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_results, f, indent=4, ensure_ascii=False)
                with open(output_md_path, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(all_md_content))
                with open(output_md_nohf_path, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(all_md_nohf_content))


                # upload output files to S3
                try:
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md", str(output_md_path), is_s3
                    )
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}_nohf.md", str(output_md_nohf_path), is_s3
                    )
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.json", str(output_json_path), is_s3
                    )
                    logging.info(f"upload from s3/oss successfully: {output_file_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to upload files to s3/oss: {str(e)}") from e

        return {
            "success": True,
            "total_pages": len(results),
            "output_s3_path": output_s3_path
        }

        # finally:
        #     # Ensure cleanup even if parser fails
        #     if os.path.exists(temp_path):
        #         os.remove(temp_path)
        #     if os.path.exists(temp_dir):
        #         os.rmdir(temp_dir)
        #     if os.path.exists(output_dir):
        #         for f in os.listdir(output_dir):
        #             os.remove(os.path.join(output_dir, f))
        #         os.rmdir(output_dir)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/image_old")
async def parse_image_old(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except TypeError:
        raise HTTPException(status_code=400, detail="Invalid filename format")
    if file_ext not in ['.jpg', '.jpeg', '.png']:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Supported: .jpg, .jpeg, .png")
    
    return await parse(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, parse_type="image")

@app.post("/parse/pdf_old")
async def parse_pdf_old(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = Form(False),
    rebuild_directory: bool = Form(False),
    describe_picture: bool = Form(False)
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except TypeError:
        raise HTTPException(status_code=400, detail="Invalid filename format")
    if file_ext not in ['.pdf']:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Supported: .pdf")

    return await parse(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, parse_type="pdf", rebuild_directory=rebuild_directory, describe_picture=describe_picture)


@app.post("/parse/file_old")
async def parse_file_old(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False,
    rebuild_directory: bool = False,
    describe_picture: bool = False
):
    try:
        try:
            file_ext = Path(input_s3_path).suffix.lower()
        except TypeError:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        if file_ext == '.pdf':
            return await parse_pdf_old(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, rebuild_directory, describe_picture)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return await parse_image_old(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

TARGET_URL = "http://localhost:8000/health"
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
            media_type=response.headers.get("content-type")
        )
    except httpx.ConnectError as e:
        return JSONResponse(
            status_code=503,
            content={"success": False, "status": 503, "detail": f"Health check failed: Unable to connect to DotsOCR service. Error: {e}"}
        )
    except httpx.TimeoutException as e:
        return JSONResponse(
            status_code=504,
            content={"success": False, "status": 504,"detail": f"Health check failed: Request to DotsOCR service timed out. Error: {e}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "status": 500,"detail": f"An unexpected error occurred during health check. Error: {e}"}
        )
    

@app.get("/health")
async def health():
    return await health_check()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6008)
