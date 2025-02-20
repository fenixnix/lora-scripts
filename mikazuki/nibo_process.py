
import os
import sys
import subprocess
from typing import Optional

def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              cpu_threads: Optional[int] = 4):
    print("Running training process / 开始运行训练进程...")
    print(toml_path)
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--num_processes","4",
        "--mixed_precision","fp16",
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

    return subprocess.Popen(args, env=customize_env)