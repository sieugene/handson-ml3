import csv

def create_csv_array_of_objects(items, filename):
    headers = items[0].keys()
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        writer.writeheader()
        writer.writerows(items)
        print(f"Csv was created and writed in {filename}")

def create_simple_csv(items: list[list[str]], filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(items)
        print(f"Csv was created and writed in {filename}")