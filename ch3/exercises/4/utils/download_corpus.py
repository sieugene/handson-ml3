from pathlib import Path
from urllib import request
import tarfile

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"

FILES = [
    "20021010_easy_ham.tar.bz2",
    "20021010_hard_ham.tar.bz2",
    "20021010_spam.tar.bz2",
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
    "20050311_spam_2.tar.bz2",
]


def download_corpus(dataset_dir="data"):
    dataset_dir = Path(dataset_dir)
    downloads_dir = dataset_dir / "downloads"

    downloads_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        archive_path = downloads_dir / filename
        if Path(archive_path).exists():
            print(f"Already downloaded, skip {filename}...")
            continue
    
        print(f"Downloading {filename}...")
        request.urlretrieve(url, archive_path)

        print(f"Extracting {filename}...")
        with tarfile.open(archive_path) as tar:
            tar.extractall(downloads_dir)

    print("download_corpus is done!")