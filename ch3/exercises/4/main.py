from utils.download_corpus import download_corpus
from utils.email_parser_rfc5322 import EmailParser
from glob import glob
from os import path

def main():
    download_corpus()
    files = {"ham": ["easy_ham", "easy_ham_2", "hard_ham"], "spam": ["spam", "spam_2"]}
    result = {}
    for label, folders in files.items():
        result[label] = 0
        for file in files[label]:
            folder = path.join("data", "downloads", file)
            count = len(glob(f"{folder}/*"))
            result[label] += count
    print(result)


main()
