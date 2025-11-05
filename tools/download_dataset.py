"""
Download a file from Google Drive by file ID or share link.
Usage:
    python tools/download_dataset.py --url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing" --dest dataset/agriculture_dataset.csv

This script handles large files by following the Google Drive confirmation token flow.
"""
import argparse
import os
import re
import requests
from urllib.parse import urlparse

CHUNK_SIZE = 32768


def get_file_id_from_url(url):
    # Common Google Drive URL formats
    # https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    # https://drive.google.com/open?id=FILE_ID
    parsed = urlparse(url)
    if 'drive.google.com' not in parsed.netloc:
        return None
    m = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if m:
        return m.group(1)
    q = dict([p.split('=') for p in parsed.query.split('&') if '=' in p])
    return q.get('id')


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)


def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def main():
    parser = argparse.ArgumentParser(description='Download a Google Drive file by URL or ID')
    parser.add_argument('--url', help='Google Drive share URL', default=None)
    parser.add_argument('--id', help='Google Drive file ID', default=None)
    parser.add_argument('--dest', help='Destination path', required=True)
    args = parser.parse_args()

    file_id = args.id
    if args.url and not file_id:
        file_id = get_file_id_from_url(args.url)

    if not file_id:
        print('Could not determine Google Drive file ID. Provide --id or a valid --url')
        return

    print(f'Downloading file id {file_id} to {args.dest} ...')
    try:
        download_file_from_google_drive(file_id, args.dest)
        print('Download completed.')
    except Exception as e:
        print('Download failed:', e)


if __name__ == '__main__':
    main()
