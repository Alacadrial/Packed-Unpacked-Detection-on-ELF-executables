import os
import requests
from bs4 import BeautifulSoup
import sys
from file_manager import File_Manager

# If packer is passed as a command line argument pass it into File_Manager() constructor else do not pass anything.
file_manager = File_Manager(sys.argv[1]) if len(sys.argv) == 2 else File_Manager()

# Site that is used to gather binary executables.
referer = "https://crackmes.one"
search_url = "https://crackmes.one/lasts/"
# Header is used to mask the program as a regular browser browsing the internet. 
header = {'User-agent': 'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.16 Safari/537.36'}
links = []

# Gathering of download links.
for i in range(1, 2):
	print("Collecting links from page: ", search_url+str(i))
	r = requests.get(search_url+str(i), headers=header)
	soup = BeautifulSoup(r.text, "html.parser")
	table = soup.find(id="content-list")
	for row in table.find_all():
		match = row.find("td")
		if match:
			link = f"{referer}/static{match.find('a')['href']}.zip"
			links.append(link)

# Downloading using the gathered links.
file_manager.create_directory("Downloads")
for index, link in enumerate(links):
	print("Downloading: ", link)
	request = requests.get(link, headers=header)
	with open(f"./Downloads/{index}.zip", "wb") as f:
		f.write(request.content) 

# Extracting downloaded zip files.
file_manager.create_directory("./Extracted/")


for file in os.listdir("./Downloads/"):
	file_manager.extract_zip(f"./Downloads/{file}", "./Extracted/", passwd="crackmes.one")

# Removing everything except executable binary files.
file_manager.delete("./Downloads/")
for file in os.listdir("./Extracted/"):
	if not file_manager.is_binary_executable(f"./Extracted/{file}"):
		file_manager.delete(f"./Extracted/{file}")

# Renaming the folder to "./Not Packed" and then making a copy of it called "./Packed"
file_manager.rename("./Extracted/", "./Not Packed")
file_manager.copy("./Not Packed", "Packed")


# Packing every file in the new directory.
for file in os.listdir("./Packed/"):
	# If a binary is failed to get packed we remove it from ./Packed directory.
	if not file_manager.pack_file(f"./Packed/{file}"):
		file_manager.delete(f"./Packed/{file}")
		
# Now we have both Packed and Unpacked versions of same executables.

