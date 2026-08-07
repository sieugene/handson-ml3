from utils.download_corpus import download_corpus
from utils.email_parser_rfc5322 import EmailParser
from utils.create_csv import create_csv_array_of_objects, create_simple_csv
from pathlib import Path


def bootstrap():
    download_corpus()

    folder_cats = {
        "ham": ["easy_ham", "easy_ham_2", "hard_ham"],
        "spam": ["spam", "spam_2"],
    }
    emails = []
    errors = []

    for folderCat, _ in folder_cats.items():
        isSpam = folderCat == "spam"
        for folderName in folder_cats[folderCat]:
            folder = Path("data", "downloads", folderName)
            files = [x for x in folder.iterdir() if x.is_file()]
            for file in files:
                try:
                    email = EmailParser(file, isSpam)
                    emails.append(email.getEmail())
                except:
                    errors.append(file)

    print(len(emails), len(errors))
    create_csv_array_of_objects(emails, "dataset.csv")
    create_simple_csv([[str(p)] for p in errors], "errors.csv")

