from pathlib import Path
from typing import TypedDict


class Header(TypedDict):
    key: str
    values: list[str]


class EmailParser:
    def __init__(self, filepath, isSpam):
        self.isSpam = isSpam
        self.filepath = filepath
        self.lines = []
        self.body_start_index = None

        self.lines = self.__read().read_text(encoding='utf-8', errors="ignore").splitlines()
        for i, line in enumerate(self.lines):
            # empty line marks the start of mail content
            if line == "":
                self.body_start_index = i + 1
                break

        self.headers = self.__parse_headers()

    def __read(self):
        file_path = Path(self.filepath)
        if file_path.exists():
            return file_path
        else:
            print("File by path", self.filepath, "doesn't exist")
            return None

    def __parse_headers(self) -> list[Header]:
        headers: list[Header] = []
        parts = self.lines[: self.body_start_index]

        for _, part in enumerate(parts):
            if not part:
                continue
            if part.startswith((" ", "\t")):
                if headers:
                    headers[-1]["values"][0] += "\n" + part
                continue
            key, *values = part.split(":", 1)
            if len(key.split()) == 1:
                headers.append({"key": key, "values": values})
        return headers

    def getKeys(self):
        keys = []
        for header in self.headers:
            keys.append(header["key"])
        return keys

    def getData(self, key):
        headers = self.__parse_headers()
        findValues: list[str] | None = None

        for header in headers:
            if header["key"] == key:
                findValues = header["values"]
                break

        if not findValues:
            return None

        return findValues

    def parse_body(self):
        body_lines = self.lines[self.body_start_index :]
        body = "\n".join(body_lines)
        return body

    def getEmail(self):
        return {
            "body": self.parse_body(),
            "from": self.getData("From"),
            "to": self.getData("To"),
            "subject": self.getData("Subject"),
            "headers": self.headers,
            "isSpam": self.isSpam
        }
