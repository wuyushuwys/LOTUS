import csv

class CSVWriter:

    def __init__(self, path, header) -> None:
        assert isinstance(header, list), f"Expected list header but got {type(header)}"
        with open(path, 'w') as f:
            csv.writer(f).writerow(header) 
        self.path = path

    def write(self, content):
        assert isinstance(content, list), f"Expected list header but got {type(content)}"
        with open(self.path, 'a') as f:
            csv.writer(f).writerow(content)