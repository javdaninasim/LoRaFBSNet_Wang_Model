
#!/usr/bin/env python3
"""
VS Code -> GitHub -> Kaggle GPU

Workflow:
    1. Validate Git repository
    2. Push current branch to GitHub
    3. Generate Kaggle bootstrap script
    4. Generate Kaggle metadata
    5. Push and run Kaggle kernel
    6. Monitor execution status
    7. Display execution logs on failure

Results remain on Kaggle.
"""

import argparse
import json
import re
import subprocess
import sys
import time

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

KAGGLE_ID = "javdaninasim/lorafb-snet-experiments"
KAGGLE_TITLE = "LoRaFB-SNet Experiments"

PROJECT_ENTRYPOINT = "main.py"
REQUIREMENTS_FILE = "requirements.txt"

DEPLOY_DIR = ROOT / ".kaggle_deploy"

KAGGLE_TIMEOUT = 7200

POLL_INTERVAL = 30
STATUS_WAIT_TIMEOUT = 7800

GPU_ENABLED = True
INTERNET_ENABLED = True


METADATA_PATH = DEPLOY_DIR / "kernel-metadata.json"

# ============================================================
# COMMAND UTILITIES
# ============================================================

def run_command(command, cwd=None, capture=False):

    print("\n>>>", " ".join(map(str, command)), flush=True)

    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture
    )


def git_output(*args):

    result = run_command(
        ["git", *args],
        cwd=ROOT,
        capture=True
    )

    return result.stdout.strip()


def check_kaggle_cli():

    print("\nChecking Kaggle CLI...")

    run_command([
        "kaggle",
        "--version"
    ])


# ============================================================
# GITHUB
# ============================================================

def get_github_repository():

    url = git_output("remote", "get-url", "origin")

    if url.startswith("git@github.com:"):

        url = url.replace(
            "git@github.com:",
            "https://github.com/",
            1
        )

    elif url.startswith("ssh://git@github.com/"):

        url = url.replace(
            "ssh://git@github.com/",
            "https://github.com/",
            1
        )

    if not url.startswith("https://github.com/"):

        raise ValueError(
            f"Invalid GitHub origin: {url}"
        )

    return url.removesuffix(".git") + ".git"


def check_git():

    print("\n[1/7] Checking Git repository...")

    branch = git_output(
        "branch",
        "--show-current"
    )

    if not branch:

        raise RuntimeError(
            "Detached HEAD detected. Checkout a branch first."
        )

    changes = git_output(
        "status",
        "--porcelain",
        "--untracked-files=normal"
    )

    if changes:

        print(changes)

        raise RuntimeError(
            "Uncommitted changes detected.\n"
            "Commit all changes before deployment."
        )

    repo_url = get_github_repository()

    if not PROJECT_ENTRYPOINT:

        raise RuntimeError("Invalid project entrypoint.")

    entrypoint = ROOT / PROJECT_ENTRYPOINT

    if not entrypoint.is_file():

        raise FileNotFoundError(entrypoint)

    print("Repository:", repo_url)
    print("Branch:", branch)

    return repo_url, branch


def push_to_github(branch):

    print("\n[2/7] Pushing to GitHub...")

    run_command([
        "git",
        "push",
        "-u",
        "origin",
        branch
    ], cwd=ROOT)

    print("GitHub push successful.")


# ============================================================
# BOOTSTRAP
# ============================================================

def create_bootstrap(repo_url, branch):

    print("\n[3/7] Generating Kaggle bootstrap...")

    bootstrap = r'''
import os
import sys
import runpy
import subprocess
from pathlib import Path


REPO_URL = __REPO_URL__
BRANCH = __BRANCH__

ENTRYPOINT = __ENTRYPOINT__
REQUIREMENTS = __REQUIREMENTS__

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
'''

    bootstrap = (
        bootstrap
        .replace("__REPO_URL__", repr(repo_url))
        .replace("__BRANCH__", repr(branch))
        .replace("__ENTRYPOINT__", repr(PROJECT_ENTRYPOINT))
        .replace("__REQUIREMENTS__", repr(REQUIREMENTS_FILE))
    )

    DEPLOY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    bootstrap_path = DEPLOY_DIR / "main.py"

    bootstrap_path.write_text(
        bootstrap,
        encoding="utf-8"
    )

    print("Bootstrap:", bootstrap_path)


# ============================================================
# METADATA
# ============================================================

def validate_metadata():

    print("\nValidating persistent Kaggle metadata...")

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_fields = [
        "id",
        "title",
        "code_file",
        "language",
        "kernel_type"
    ]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(
                f"Missing metadata field: {field}"
            )

    if metadata["id"] != KAGGLE_ID:
        raise ValueError(
            "Kaggle kernel ID mismatch."
        )

    if metadata["kernel_type"] != "script":
        raise ValueError(
            "Expected kernel_type='script'."
        )

    code_path = DEPLOY_DIR / metadata["code_file"]

    if not code_path.is_file():
        raise FileNotFoundError(
            f"Kernel source not found: {code_path}"
        )

    print("Metadata validated successfully.")

    print("Kernel ID:", metadata["id"])
    print("Code file:", metadata["code_file"])
    print("GPU:", metadata.get("enable_gpu"))
    print("Internet:", metadata.get("enable_internet"))

    return metadata


# ============================================================
# KAGGLE DEPLOYMENT
# ============================================================

def deploy_to_kaggle(no_run=False):

    print("\n[5/7] Deploying Kaggle kernel...")

    command = [

        "kaggle",
        "kernels",
        "push",

        "-p",
        str(DEPLOY_DIR),

        "--timeout",
        str(KAGGLE_TIMEOUT)

    ]

    if no_run:

        command.append("--no-run")

    run_command(command)

    print("\nKaggle deployment submitted.")


# ============================================================
# EXECUTION MONITORING
# ============================================================

def get_kernel_status():

    result = subprocess.run(

        [
            "kaggle",
            "kernels",
            "status",
            KAGGLE_ID
        ],

        text=True,
        capture_output=True
    )

    output = (
        result.stdout + "\n" + result.stderr
    ).strip()

    print("\nKaggle status:", output, flush=True)

    if result.returncode != 0:

        return "unknown"

    text_status = output.lower()

    # Match actual status labels, avoiding substring
    # matches such as "incomplete".

    match = re.search(
        r'\b(queued|running|complete|completed|error|failed|cancelled)\b',
        text_status
    )

    if not match:

        return "unknown"

    status = match.group(1)

    if status in ("complete", "completed"):
        return "complete"

    if status in ("error", "failed", "cancelled"):
        return "failed"

    return status


def show_logs():

    print("\nRetrieving Kaggle execution logs...")

    subprocess.run(
        [
            "kaggle",
            "kernels",
            "logs",
            KAGGLE_ID
        ],
        check=False
    )


def wait_for_completion():

    print("\n[6/7] Monitoring Kaggle execution...")

    start_time = time.monotonic()

    while True:

        elapsed = time.monotonic() - start_time

        if elapsed > STATUS_WAIT_TIMEOUT:

            raise TimeoutError(
                "Kaggle monitoring timeout reached."
            )

        status = get_kernel_status()

        if status == "complete":

            print("\nKaggle execution completed successfully.")

            return

        if status == "failed":

            show_logs()

            raise RuntimeError(
                "Kaggle execution failed. Check execution logs."
            )

        if status in ("queued", "running", "unknown"):

            print(
                f"Waiting... {int(elapsed)} seconds elapsed.",
                flush=True
            )

            time.sleep(POLL_INTERVAL)


# ============================================================
# FINAL REPORT
# ============================================================

def final_report():

    print("\n[7/7] Deployment report")

    print("=" * 70)

    print("Project:", KAGGLE_TITLE)

    print("Kernel:", KAGGLE_ID)

    print(
        "URL:",
        f"https://www.kaggle.com/code/{KAGGLE_ID}"
    )

    print("Results: Stored on Kaggle")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="LoRaFB-SNet Kaggle deployment"
    )

    parser.add_argument(
        "--no-run",
        action="store_true"
    )

    parser.add_argument(
        "--skip-git-push",
        action="store_true"
    )

    parser.add_argument(
        "--no-wait",
        action="store_true"
    )

    args = parser.parse_args()

    try:

        check_kaggle_cli()

        repo_url, branch = check_git()

        if not args.skip_git_push:

            push_to_github(branch)

        create_bootstrap(
            repo_url,
            branch
        )

        validate_metadata()

        deploy_to_kaggle(
            no_run=args.no_run
        )

        if args.no_run:

            print("\nKernel uploaded without execution.")

            return

        if not args.no_wait:

            wait_for_completion()

        final_report()

    except subprocess.CalledProcessError as error:

        print(
            "\nCommand failed:",
            error.returncode
        )

        sys.exit(error.returncode)

    except KeyboardInterrupt:

        print("\nMonitoring interrupted by user.")

        sys.exit(130)

    except Exception as error:

        print("\nDEPLOYMENT ERROR:", error)

        sys.exit(1)


if __name__ == "__main__":

    main()
