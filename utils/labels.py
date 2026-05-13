import csv


def load_labels(label_path):
    with open(label_path, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]
