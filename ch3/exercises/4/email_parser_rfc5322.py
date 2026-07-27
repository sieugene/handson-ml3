
# RFC 5322 format

from pathlib import Path


# Another way use a pyhon default lib
# import email
# msg = email.message_from_file(open("./data/downloads/easy_ham/00001.7c53336b37003a9286aba55d2945844c", "r"))
# print(msg['Subject'])
# print(msg["From"])
# print(msg["To"])
# print(msg.get_payload())

class EmailParser:
    def __init__(self, filepath):
        self.filepath = filepath

    def read(self):
        file_path = Path(self.filepath)
        if file_path.exists():
              return file_path
        else:
            print("File by path", self.filepath , "doesn't exist")
            return None

    def parse_headers(self):
        ...

    def parse_body(self):
        file = self.read()
        if file:
            content = file.read_text()
            lines = content.splitlines()
            empty_line_index = None
            for i, line in enumerate(lines):
                # empty line marks the start of mail content
                if line == '':
                    empty_line_index = i
                    break

            body_start_index = empty_line_index + 1
            # Take all content from the pointer
            body_lines = lines[body_start_index:]
            body = "\n".join(body_lines)
            return body
        else:
            print("File by path", self.filepath , "doesn't exist")
            return None
        

email = EmailParser("./data/downloads/easy_ham/00001.7c53336b37003a9286aba55d2945844c")
print(email.parse_body())
