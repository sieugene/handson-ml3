
from pathlib import Path
class EmailParser:
	def __init__(self, filepath):
		self.filepath = filepath
		self.lines = []
		self.body_start_index = None

		self.lines = self.read().read_text().splitlines()
		for i, line in enumerate(self.lines):
			# empty line marks the start of mail content
			if line == '':
				self.body_start_index = i + 1
				break

	def read(self):
		file_path = Path(self.filepath)
		if file_path.exists():
			return file_path
		else:
			print("File by path", self.filepath , "doesn't exist")
			return None

	def parse_headers(self):
		headers = []
		parts = self.lines[:self.body_start_index]

		for _, part in enumerate(parts):
			if not part:
				continue
			if part.startswith((" ", "\t")):
				if headers:
					headers[-1]["values"][0] = "\n" + part
				continue
			key, *values = part.split(":")
			if len(key.split()) == 1:
				headers.append({"key": key, "values": values})
		return headers

	def getKeys(self):
		headers = self.parse_headers()
		keys = []
		for header in headers:
			keys.append(header["key"])
		return keys
	
	def parse_body(self):
		body_lines = self.lines[self.body_start_index:]
		body = "\n".join(body_lines)
		return body

		

# emailCustom = EmailParser("./data/downloads/easy_ham/00001.7c53336b37003a9286aba55d2945844c")
# print(emailCustom.parse_body())
# print(emailCustom.getKeys())