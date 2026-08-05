from utils.download_corpus import download_corpus
from utils.email_parser_rfc5322 import EmailParser
from glob import glob
from pathlib import Path

def main():
    download_corpus()

    folder_cats = {"ham": ["easy_ham", "easy_ham_2", "hard_ham"], "spam": ["spam", "spam_2"]}
    emails = []

    for folderCat, _ in folder_cats.items():
        for folderName in folder_cats[folderCat]:
            folder = Path("data", "downloads", folderName)
            files = [x for x in folder.iterdir() if x.is_file()]
            for file in files:
                email = EmailParser(file)
                emails.append(email.getEmail())

    print(emails)

main()
