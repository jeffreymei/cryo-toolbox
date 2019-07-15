'''
Call this with "python parse_nsidc.py -u https://directory/with/files/ -e ext"
note the URL must end with "/", and the extension should just be 'xml', 'hdf' and not have '.'
This will save all .ext files from the URL in folders in the current directory.
If you don't have a .netrc file with permissions, (https://nsidc.org/support/faq/what-options-are-available-bulk-downloading-data-https-earthdata-login-enabled), you will need to add a username, with --user.
'''
from lxml import html
import requests
import sys, os, argparse

parser = argparse.ArgumentParser(description="This downloads *.ext files from a NSIDC server. Make sure you have a login to EarthData, either with a .netrc file, or you can just pass --user <USERNAME> as an argument (it will prompt for a password).")
parser.add_argument('-e', '--extension', help='Extension, e.g. xml', required=True)
parser.add_argument('-u', '--url', help='URL where files are; must be a directory', required=True)
parser.add_argument('-n', '--user', help='Username for Earthdata login (use this if you do not have stored cached credentials in a .netrc file)', required=False)
args = parser.parse_args()

try: assert args.url[-1] == '/'
except: raise Exception("URL must be directory ending in /")
try: assert len(args.extension)<5
except: "extension should be 'html', 'jpg' etc"




page = requests.get(args.url)
tree = html.fromstring(page.content)
out = tree.xpath('//td[@class="indexcolname"]//a/@href')[1:]
filtered_items = [i for i in out if '.'+args.extension in i]

print("Getting " + "."+args.extension + " files from " + args.url)

with open("./filenames.txt", "w") as f:
	for item in filtered_items:
		f.write(args.url+'%s\n' % item)
	# for item in outjpg:
	# 	f.write('https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IODMS1B.001/2011.10.25/'+'%s\n' % item)
if len(sys.argv) == 5:
	command = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r -l1 -np -e robots=off -i ./filenames.txt'
else:
	command = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r -l1 -np -e robots=off -i ./filenames.txt --http-user='+args.user+' --ask-password'
os.system(command)
os.remove("./filenames.txt")
