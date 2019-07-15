'''
Call this with "python parse_nsidc https://directory/with/files/ ext"
note the URL must end with "/", and the extension should just be 'xml', 'hdf' and not have '.'
This will save all .ext files from the URL in folders in the current directory.
If you don't have a .netrc file with permissions, (https://nsidc.org/support/faq/what-options-are-available-bulk-downloading-data-https-earthdata-login-enabled)
please enter your username as a 3rd argument, e.g. 'python https://url/folder/ ext johnsmith'
'''
from lxml import html
import requests
import sys, os
if len(sys.argv) == 3:
	URL = str(sys.argv[1])
	extension = str(sys.argv[2])
	try: assert URL[-1] == '/'
	except: raise Exception("URL must be directory")
	try: assert len(extension)<5
	except: "extension should be 'html', 'jpg' etc"
elif len(sys.argv) == 4:
	URL = str(sys.argv[1])
	extension = str(sys.argv[2])
	try: assert URL[-1] == '/'
	except: raise Exception("URL must be directory")
	try: assert len(extension)<5
	except: "extension should be 'html', 'jpg' etc"
	username = sys.argv[3]
else:
	print("Need two arguments, https://folder/that/contains/files and extensions to download, with an optional 3rd argument (username)")
	# sys.exit()



page = requests.get(URL)
tree = html.fromstring(page.content)
out = tree.xpath('//td[@class="indexcolname"]//a/@href')[1:]
filtered_items = [i for i in out if '.'+extension in i]

print("Getting " + "."+extension + " files from " + URL)

with open("./filenames.txt", "w") as f:
	for item in filtered_items:
		f.write(URL+'%s\n' % item)
	# for item in outjpg:
	# 	f.write('https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IODMS1B.001/2011.10.25/'+'%s\n' % item)
if len(sys.argv) == 3:
	command = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r -l1 -np -e robots=off -i ./filenames.txt'
else:
	command = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r -l1 -np -e robots=off -i ./filenames.txt --http-user='+username+' --ask-password'
os.system(command)
