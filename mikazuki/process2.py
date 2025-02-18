
import asyncio
import os
import sys
import subprocess
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm
from mikazuki.launch_utils import base_dir_path


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]
            if sys.platform == "win32":
                customize_env["USE_LIBUV"] = "0"
                args[3:3] = ["--rdzv_backend", "c10d"]

    subprocess.Popen(args, env=customize_env)