#!/usr/bin/python

import json
import os.path
import re
import requests
import string
import sys
import urllib
import datetime

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
	needToFullDownload = False
	if not os.path.isfile('Downloads_2/dateDigitized.txt'):
		needToFullDownload = True
		recent_date = datetime.datetime(1000, 1, 1, 0, 0, 0)
	else:
		read_rec = open('Downloads_2/dateDigitized.txt', 'r')
		rec_date = read_rec.read()
		rec_date = datetime.datetime.strptime(rec_date,"%Y-%m-%dT%H:%M:%S")
		recent_date = rec_date
		read_rec.close()
	
	if not os.path.isfile('Downloads_2/TrainData.txt'):
		os.makedirs('Downloads_2')
	
	td = open('Downloads_2/TrainData.txt', 'w')	
	    
	while current <= total_pages:
		url = url_for_FSA.format(current, count)
		print(url)
		current += 1
		response_for_FSA = requests.get(url, headers={'Authorization ':'Token token=' + token}).json()
		result_list = response_for_FSA['nyplAPI']['response']
		total_pages = int(response_for_FSA['nyplAPI']['request']['totalPages'])
		for i in range(len(result_list['capture'])):
			if result_list['capture'][i]['typeOfResource'] == 'still image':
				if needToFullDownload :
					print("needToFullDownload is true")
					capture_id = str(result_list['capture'][i]['imageID'])
					captures.append(capture_id)
					titles.append(str(result_list['capture'][i]['title']))
					

					
					title = ("Downloads_2/Trial_{0}").format(id)
					print("folder title will be '"+title+"'")
					id += 1
					#Create folder based on the item title
					if not os.path.exists(title):
						os.makedirs(title)

					# dateDigitized recording the recent most uuid apdated with the ALT Text
					curr_date = datetime.datetime.strptime(str(result_list['capture'][i]['dateDigitized']),"%Y-%m-%dT%H:%M:%SZ")
					if recent_date < curr_date:
						recent_date = curr_date
						rec = open('Downloads_2/dateDigitized.txt', 'w')
						rec.write(str(recent_date.isoformat()))
						print(capture_id, "dateDigitized", recent_date.isoformat())
						rec.close()

        
					if not os.path.isfile(title+'/'+'Readme.txt'):
						f = open(title+'/'+'Readme.txt', 'w')
						f.write(str(result_list['capture'][i]['title']))
						f.close()

					if not os.path.isfile(title + '/' + capture_id + deriv_type + '.jpg'):
						try:
							urllib.request.urlretrieve(img_url_base + capture_id + '&t='+deriv_type,title + '/' + capture_id + deriv_type + '.jpg')
							print(capture_id, deriv_type, "of")
                            					
							# create training data
							td.write(title + '/' + capture_id + deriv_type + '.jpg#0\t' + str(result_list['capture'][i]['title']) + '\n')
					
						except Exception as e:
							print(e)
							continue
				
				else:               
					print("needToFullDownload is false")                
					# dateDigitized recording the recent most uuid apdated with the ALT Text
					curr_date = datetime.datetime.strptime(str(result_list['capture'][i]['dateDigitized']),"%Y-%m-%dT%H:%M:%SZ")
			
					print(rec_date)
					if rec_date < curr_date:
						if recent_date < curr_date:
							recent_date = curr_date
							rec = open('Downloads_2/dateDigitized.txt', 'w')
							rec.write(str(recent_date.isoformat()))
							print("dateDigitized", recent_date.isoformat())
							rec.close()
						
						capture_id = str(result_list['capture'][i]['imageID'])
						captures.append(capture_id)
						titles.append(str(result_list['capture'][i]['title']))
                        
						today = datetime.datetime.now()
						today = today.strftime("%d-%m-%Y_%H-%M-%S")
						title = ("Downloads_2/Trial_{1}_{0}").format(id,today)
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
								# create training data
								td.write(title + '/' + capture_id + deriv_type + '.jpg#0\t' + str(result_list['capture'][i]['title']) + '\n')
							except Exception as e:
								print(e)
								continue
						
	td.close()

	
if __name__ == "__main__":
	main()