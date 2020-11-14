#!/usr/bin/python

import json
import os.path
import re
import requests
import string
import sys
import urllib

api_base = 'http://api.repo.nypl.org/api/v1/'
img_url_base = "http://images.nypl.org/index.php?id="
url_for_FSA = 'http://api.repo.nypl.org/api/v1/items/e5462600-c5d9-012f-a6a3-58d385a7bc34?withTitles=yes&page={0}&per_page={1}'
token = 'gy5zj18sf99ddtxn'
deriv_type = 'r'.lower()


def main():
	current = 1
	total_pages = 1
	count = 500
	imageId = set()
	captures = []
	titles= []
	id = 1
	while current <= total_pages:
		url = url_for_FSA.format(current, count)
		print(url)
		current += 1
		response_for_FSA = requests.get(url, headers={'Authorization ':'Token token=' + token}).json()
		print("here")
		result_list = response_for_FSA['nyplAPI']['response']
		total_pages = int(response_for_FSA['nyplAPI']['request']['totalPages'])
		for i in range(len(result_list['capture'])):
			if result_list['capture'][i]['typeOfResource'] == 'still image':
				capture_id = str(result_list['capture'][i]['imageID'])
				captures.append(capture_id)
				titles.append(str(result_list['capture'][i]['title']))

				title = ("Downloads/Trial_{0}").format(id)
				print("folder title will be '"+title+"'")
				id += 1
				#Create folder based on the item title
				if not os.path.exists(title):
					os.makedirs(title)
		
				if not os.path.isfile(title+'/'+'Readme.txt'):
					f = open(title+'/'+'Readme.txt', 'w')
					f.write(str(result_list['capture'][i]['title']))
					f.close()

				if not os.path.isfile(title + '/' + capture_id + deriv_type + '.jpg'):
					try:
						urllib.request.urlretrieve(img_url_base + capture_id + '&t='+deriv_type,title + '/' + capture_id + deriv_type + '.jpg')
						print(capture_id, deriv_type, "of")
					except Exception as e:
						print(e)
						continue

	
if __name__ == "__main__":
	main()