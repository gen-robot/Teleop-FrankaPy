import os
import tyro
import shutil
import requests
import zipfile
from tqdm import tqdm
from dataclasses import dataclass

try: 
    from frankapy import FRANKAPY_PATH
except:
    FRANKAPY_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROBOT_URLS = {
    "panda": "https://cloud.tsinghua.edu.cn/f/402cfb147de84d1990d0/?dl=1",
    "cobot": "example",
}

ROBOT_CHECK_FILES = {
    "panda": "panda_v3.urdf",
    "cobot": "cobot.urdf",
}


@dataclass
class Args():
    robot_name: str
    assets_base_dir: str = os.path.join(FRANKAPY_PATH, "assets")    
    

def safe_rmtree(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def safe_remove(path: str):
    if os.path.isfile(path):
        os.remove(path)


def get_dir_size(path: str) -> int:
    """Calculates the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def check_and_download_assets(robot_name: str = "panda", assets_base_dir: str = os.path.join(FRANKAPY_PATH, "assets")):
    """
    Check and download the assets directory for the specified robot.

    Args:
        robot_name (str): Robot name, e.g., 'panda' or 'cobot'
        assets_base_dir (str): Base directory for storing assets, default is ./frankapy/assets

    Returns:
        str: Final robot directory path, e.g., ./frankapy/assets/panda
    """
    if robot_name not in ROBOT_URLS:
        raise ValueError(f"Unknown robot '{robot_name}', please add the corresponding download URL in ROBOT_URLS")

    assets_target_dir = os.path.join(assets_base_dir, robot_name)
    tmp_dir = "./tmp_assets"
    tmp_zip_path = os.path.join(tmp_dir, f"{robot_name}.zip")
    tmp_extract_dir = os.path.join(tmp_dir, f"{robot_name}")

    # Step 1: Check if target directory already exists and is valid
    if os.path.isdir(assets_target_dir):
        dir_size_bytes = get_dir_size(assets_target_dir)
        # Check if size is greater than 1 MB (1 * 1024 * 1024 bytes)
        if dir_size_bytes > 1048576 and os.path.isfile(os.path.join(assets_target_dir, ROBOT_CHECK_FILES[robot_name])):
            dir_size_mb = dir_size_bytes / (1024 * 1024)
            print(f"[✓] Directory found and seems valid (size: {dir_size_mb:.2f} MB): {assets_target_dir}")
            return assets_target_dir
        else:
            dir_size_mb = dir_size_bytes / (1024 * 1024)
            print(f"[!] Directory found but seems incomplete (size: {dir_size_mb:.2f} MB). Deleting and re-downloading.")
            safe_rmtree(assets_target_dir)

    # Ensure base directory exists
    os.makedirs(assets_base_dir, exist_ok=True)

    url = ROBOT_URLS[robot_name]

    try:
        # Step 2: Download zip file
        print(f"[↓] {assets_target_dir} not found, starting download from {url}")

        # Clean up and recreate temporary directory before downloading
        safe_rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        with open(tmp_zip_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {robot_name}.zip"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"[✓] Download completed: {tmp_zip_path}")

        # Step 3: Extract to temporary directory
        print(f"[↓] Starting extraction to temporary directory {tmp_extract_dir}")

        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_extract_dir)
        print(f"[✓] Extraction completed")

        # Step 4: Find assets/<robot_name> directory and move it
        robot_src = None
        for root, dirs, files in os.walk(tmp_extract_dir):
            if robot_name in dirs:
                robot_src = os.path.join(root, robot_name)
                break

        if robot_src is None:
            raise FileNotFoundError(f"{robot_name} directory not found after extraction, please check the archive structure")

        shutil.move(robot_src, assets_target_dir)
        print(f"[✓] Moved {robot_name} to {assets_target_dir}")

    except Exception as e:
        print(f"[✗] Error: {e}")
        # Clean up all residues
        safe_rmtree(tmp_dir)
        safe_rmtree(assets_target_dir)
        raise RuntimeError(f"Failed to download or install {robot_name}, please retry") from e

    finally:
        # Clean up temporary directory whether successful or failed
        safe_rmtree(tmp_dir)

    return assets_target_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    check_and_download_assets(args.robot_name, args.assets_base_dir)
