import asyncio
import logging
import os
import signal
from multiprocessing import Process
from typing import List

from configs.settings import Settings
from resources.functions import get_resource_manager


def init_worker_sync(settings: Settings, process_index: int):
    from resources.resource_manager import GlobalResourceManager

    logging.info(f"[WORKER: {os.getpid()}] Worker process started")
    load_dotenv(override=True, dotenv_path=".env")
    os.environ["OFNIL_WORKER_PROCESS"] = "true"
    settings.shared_variables.pid_index_mappings[os.getpid()] = process_index
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    GlobalResourceManager._instance = None
    asyncio.run(init_worker(settings))


async def init_worker(settings: Settings):
    await get_resource_manager().initialize(settings=settings)
    await asyncio.to_thread(settings.shared_variables.stop_subprocess_event.wait)
    await cleanup_worker()


async def cleanup_worker():
    await get_resource_manager().cleanup()


class ProcessManager:
    def __init__(self, settings: Settings) -> None:
        self.num_workers: int = settings.num_worker_processes
        self.process_list: List[Process] = []
        for i in range(self.num_workers):
            process: Process = Process(target=init_worker_sync, args=(settings, i))
            process.start()
            self.process_list.append(process)

    async def cleanup(self) -> None:
        try:
            await asyncio.wait_for(self._wait_for_processes(), timeout=60.0)
        except asyncio.TimeoutError:
            for process in self.process_list:
                if process.is_alive():
                    process.kill()
            await asyncio.gather(self._wait_for_processes())

    async def _wait_for_processes(self) -> None:
        tasks = [asyncio.to_thread(process.join) for process in self.process_list]
        await asyncio.gather(*tasks)
