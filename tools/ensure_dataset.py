"""
Ensure dataset is present locally. If found in archive/, copy it. Otherwise download from Google Drive.
Usage:
    python tools/ensure_dataset.py --dest dataset/agriculture_dataset.csv --url <drive-url>

This avoids redownloading if a copy exists in archive/ or locally.
"""
import argparse
import os
import shutil
from pathlib import Path

try:
    from tools.download_dataset import get_file_id_from_url, download_file_from_google_drive
except Exception:
    # If module import fails because of packaging, attempt relative import
    from download_dataset import get_file_id_from_url, download_file_from_google_drive


def ensure_dataset(dest, url=None, archive_path=None):
    dest_path = Path(dest)
    if dest_path.exists():
        print(f"Dataset already exists at {dest}")
        return True

    # If archive path provided or default archive, try to copy
    if not archive_path:
        archive_path = Path(__file__).resolve().parents[1] / 'archive' / dest_path.name
    else:
        archive_path = Path(archive_path)

    if archive_path.exists():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying dataset from archive: {archive_path} -> {dest}")
        shutil.copy2(str(archive_path), str(dest_path))
        print("Copy complete")
        return True

    # If no archive copy, attempt download if URL provided
    if url:
        file_id = get_file_id_from_url(url)
        if not file_id:
            print("Could not parse file ID from provided URL")
            return False
        print(f"Downloading dataset from Google Drive (id={file_id}) to {dest}")
        try:
            download_file_from_google_drive(file_id, str(dest_path))
            print("Download complete")
            return True
        except Exception as e:
            print("Download failed:", e)
            return False

    print("Dataset not found locally or in archive and no download URL provided.")
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensure dataset is present locally (copy from archive or download)')
    parser.add_argument('--dest', required=True, help='Destination path for dataset')
    parser.add_argument('--url', help='Google Drive URL to download from if needed')
    parser.add_argument('--archive', help='Explicit archive path to copy from (optional)')
    args = parser.parse_args()

    success = ensure_dataset(args.dest, url=args.url, archive_path=args.archive)
    if not success:
        exit(1)
