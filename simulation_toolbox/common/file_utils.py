import os

def file_exists(files_to_check):
	
	if isinstance(files_to_check,list):
		for filepath in files_to_check:
			file_exists(files_to_check=filepath)
	
	elif isinstance(files_to_check, str):
		if not os.path.isfile(files_to_check):
			raise Exception(f"{files_to_check} not found.")
	else:
		raise Exception(f"Only strings can be checked. {files_to_check} is not a string.")