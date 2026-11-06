
import os
import sys
import runpy
import subprocess
from pathlib import Path


REPO_URL = 'https://github.com/javdaninasim/LoRaFBSNet_Wang_Model.git'
BRANCH = 'main'

ENTRYPOINT = 'main.py'
REQUIREMENTS = 'requirements.txt'

WORK_DIR = Path("/kaggle/working")

REPO_DIR = WORK_DIR / "LoRaFBsnet_Wang"


def execute(command):

    print(
        "\n>>> " + " ".join(map(str, command)),
        flush=True
    )

    subprocess.run(
        command,
        check=True
    )


def main():

    print("=" * 70)
    print("LoRaFB-SNet | Kaggle GPU Execution")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Clone public GitHub repository
    # --------------------------------------------------------

    print("\n[1/4] Cloning GitHub repository...")

    if REPO_DIR.exists():

        import shutil
        shutil.rmtree(REPO_DIR)

    execute([
        "git",
        "clone",
        "--depth", "1",
        "--single-branch",
        "--branch", BRANCH,
        REPO_URL,
        str(REPO_DIR)
    ])

    # --------------------------------------------------------
    # 2. GPU verification
    # --------------------------------------------------------

    print("\n[2/4] Checking GPU...")

    execute([
        sys.executable,
        "-c",
        (
            "import torch; "
            "print('PyTorch:', torch.__version__); "
            "print('CUDA:', torch.cuda.is_available()); "
            "print('GPU:', "
            "torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else 'Unavailable')"
        )
    ])

    # --------------------------------------------------------
    # 3. Install dependencies
    # --------------------------------------------------------

    print("\n[3/4] Installing dependencies...")

    requirements_path = REPO_DIR / REQUIREMENTS

    if requirements_path.is_file():

        execute([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path)
        ])

    else:

        print("requirements.txt not found. Skipping.")

    # --------------------------------------------------------
    # 4. Execute project
    # --------------------------------------------------------

    print("\n[4/4] Starting main.py...")

    entrypoint = REPO_DIR / ENTRYPOINT

    if not entrypoint.is_file():

        raise FileNotFoundError(entrypoint)

    os.chdir(REPO_DIR)

    sys.path.insert(0, str(REPO_DIR))

    # Useful for unbuffered logs
    os.environ["PYTHONUNBUFFERED"] = "1"

    runpy.run_path(
        str(entrypoint),
        run_name="__main__"
    )

    print("\nProject execution completed successfully.")


if __name__ == "__main__":

    main()
