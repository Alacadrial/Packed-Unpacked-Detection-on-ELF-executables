import os
import string
import magic
from zipfile import ZipFile
from distutils.dir_util import copy_tree
from shutil import copyfile as copy_file
from shutil import rmtree as rm_tree
import subprocess

class File_Manager():
	def __init__(self, packer_path=""):
		self.packer = packer_path
		self.counter = 0

	def create_directory(self, name):
		if not os.path.exists(name):
			print(f"Creating directory: {name}.")
			os.mkdir(name)
		else:
			print(f"Directory: {name}, already exists.")

	def rename(self, source, destination):
		if os.path.exists(destination) or not os.path.exists(source):
			print(f"Cannot do renaming operation.")
		else:
			print(f"Renaming: {source}, to ---> {destination}.")
			os.rename(source, destination)

	def copy(self, file_path, copy_location):
		function = copy_tree if os.path.isdir(file_path) else copy_file
		if os.path.exists(file_path):
			print(f"Copying: {file_path}, to ----> {copy_location}.")
			function(file_path, copy_location)
		else:
			print(f"File: {file_path}, not found.")

	def delete(self, file_path):
		function = rm_tree if os.path.isdir(file_path) else os.remove
		if os.path.exists(file_path):
			print(f"Removing: {file_path}.")
			function(file_path)
		else:
			print(f"File: {file_path}, not found.")

	def extract_zip(self, file_to_extract, extraction_location="./", passwd=None):
		print(f"Extracting {file_to_extract}, to {extraction_location}.")
		try:
			with ZipFile(file_to_extract) as zip_ref:
				for file in zip_ref.namelist():
					if passwd is None:
						zip_ref.extract(file, extraction_location)
					else:
						zip_ref.extract(file, extraction_location, bytes(passwd,"utf-8"))
					#self.rename(extraction_location + file, extraction_location + str(self.counter))
					#self.counter += 1
		except Exception as err:
			print("Error while extracting: ", err)

	# Gives all printable strings in a file. Didn't have to use this function because found another way.
	def strings(self, file_path, min_len=3):
		printable_strings = list()
		catch = ""
		with open(file_path, errors="ignore") as f:
			for char in f.read():
				if char in string.printable:
					catch += char
					continue
				if len(catch) >= min_len:
					printable_strings.append(catch)
				catch = ""
		return printable_strings

	# Code determined to find if a file is a binary executable or not. Looks for magic bytes and magic keywords.
	def is_binary_executable(self, file_path):
		keywords = ["elf", "pe32", "pe64", "executable"]
		# Example description: ELF 32-bit LSB executable, Intel 80386, version 1 (SYSV), statically linked, not stripped
		# Another example: PE32 executable (console) Intel 80386 Mono/.Net assembly, for MS Windows
		if os.path.exists(file_path):
			# Magic library gives description about a file, looks for magic bytes declaring what file type a file is.
			description = magic.from_file(file_path) 
			for keyword in keywords:
				if keyword in description.lower():
					return True
		return False

	# Returns Bool, tries to pack and returns the result of the action. If packing operation was a success returns True.
	def pack_file(self, file_path):
		print("Trying to pack: ", file_path)
		try:
			packing_result = subprocess.check_output([self.packer, "--best", file_path])
			print("Packing succesful: ", file_path)
			return True
		except Exception as err:
			print("Couldn't pack: ", file_path)
			return False


def main():
	pass

if __name__ == '__main__':
	main()
