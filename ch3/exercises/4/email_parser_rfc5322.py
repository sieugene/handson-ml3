
import email
from pathlib import Path
from pprint import pprint



def parseHeaders(parts):
	headers = []
	for i, part in enumerate(parts):
		if part:
			key, *values = part.split(":", 1)
			if '\t' in key:
				try:
					headerLastIndex = len(headers) - 1
					nextHeaderValues = [*headers[headerLastIndex]["values"], key]
					headers[headerLastIndex] = {
						"key": headers[headerLastIndex]["key"],
						"values": ["\n".join(nextHeaderValues)]
					}
				except Exception as e:
					print("e", e)
			else:
				headers.append({
					"key": key,
					"values": [*values]
				})
	return headers


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

			body_end_index = empty_line_index + 1
			# Take all content before the pointer
			body_lines = lines[:body_end_index]
			original_part_before_content = "\n".join(body_lines)

			return parseHeaders(body_lines)
		else:
			print("File doesn't exist")
			return None

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
			print("File doesn't exist")
			return None
		

emailCustom = EmailParser("./data/downloads/easy_ham/00001.7c53336b37003a9286aba55d2945844c")
# print(email.parse_body())
pprint(emailCustom.parse_headers())

# Another way use a pyhon default lib
msg = email.message_from_file(open("./data/downloads/easy_ham/00001.7c53336b37003a9286aba55d2945844c", "r"))
# print(msg['References'])
# print(msg["From"])
# print(msg["Received"])
# print(msg.get_payload())
# print("lib\n",msg.keys())