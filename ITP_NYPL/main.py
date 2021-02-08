import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template,request,jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from pyngrok import ngrok
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

import json
import os.path
import re
import requests
import string
import sys
import urllib
import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
token_nizer='model/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL']=token_nizer
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

api_base = 'http://api.repo.nypl.org/api/v1/'
img_url_base = "http://images.nypl.org/index.php?id="
url_for_FSA = 'http://api.repo.nypl.org/api/v1/items/e5462600-c5d9-012f-a6a3-58d385a7bc34?withTitles=yes&page={0}&per_page={1}'
token = 'gy5zj18sf99ddtxn'
deriv_type = 'r'.lower()
def start_ngrok():
    

    url = ngrok.connect(3000).public_url
    print(' * Tunnel URL:', url)
start_ngrok()
global filesave

global file_name
file_name=''

def extract_features(filename):
	# load the model
	model = VGG16()
	# re-structure the model
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# load the photo
	image = load_img(filename, target_size=(224, 224))
	# convert the image pixels to a numpy array
	image = img_to_array(image)
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# prepare the image for the VGG model
	image = preprocess_input(image)
	# get features
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = ' '
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		if word == 'endseq':
			break
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		
	op=in_text
	return op

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/uploadimage', methods=['POST'])
def uploadimage():
	if 'files[]' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	
	files = request.files.getlist('files[]')
	
	errors = {}
	success = False
	
	for file in files:
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			img_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
			filesave=filename
			print(img_path)
			print(filesave)
			tokenizer=load(open((os.path.join(app.config['MODEL'],'tokenizer_fb.pkl')),"rb"))
			model = load_model(os.path.join(app.config['MODEL'], 'model-ep001-loss1.292-val_loss0.992.h5'))
			max_length = 16
			photo = extract_features(img_path)
			img = Image.open(img_path)

			description = generate_desc(model, tokenizer, photo, max_length)
			print(description)
			success = True
		else:
			errors[file.filename] = 'File type is not allowed'
	
	if success and errors:
		errors['message'] = 'File(s) successfully uploaded'
		resp = jsonify(errors)
		resp.status_code = 206
		return resp
	if success:
		resp = jsonify({'message' : description,'description':description,'filename':filesave})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify(errors)
		resp.status_code = 400
		return resp
	
@app.route('/change', methods=['POST'])
def change():
	data = request.get_json()

	descc = data['name']
	fname=data['fname']
	print(descc)
	print(fname)
	# with open(os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt'), 'w') as f:
	# 		f.write(str(file_name))
	with open(os.path.join(app.config['UPLOAD_FOLDER'], 'description_r.txt'), 'a+') as f:
			
			f.write(str(fname)+" "+str(descc)+'\n')
	return render_template('index.html')

@app.route('/retrain', methods=['POST'])
def retrain():

	def update_changes(filename_from, filename_to, with_desc):
		map = {}
		s = set()
		file1 = open(filename_from,'r') #_r
		file2 = open(filename_to,'r') #_fb
        
		for line2 in file2:
			if not with_desc: #Training file
				k = line2.strip()
				s.add(k)
			else: #Description file
				(k,v) = line2.split(" ", 1)
				k = k.strip()
				v = v.strip()
				map[k] = v
        
		for line1 in file1:
			if not with_desc: #Training file
				s.add(line1.strip())
			else : #Description file
				(img_id,desc) = line1.split(" ", 1)
				img_id = img_id.strip()
				desc = desc.strip()
				map[img_id] = desc
            
		file1.close()
		file2.close()
        
		if not with_desc:
            #clear both files
			open(os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt'), 'w').close()
			open(os.path.join(app.config['UPLOAD_FOLDER'], 'train_r.txt'), 'w').close()
			for k in s:#Training file
				with open(os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt'), 'a+') as f:
					f.write(k+'\n')
		else :
            #clear both files
			open(os.path.join(app.config['UPLOAD_FOLDER'], 'descriptions_fb.txt'), 'w').close()
			open(os.path.join(app.config['UPLOAD_FOLDER'], 'description_r.txt'), 'w').close()
			for k,v in map.items():#Description file
				with open(os.path.join(app.config['UPLOAD_FOLDER'], 'descriptions_fb.txt'), 'a+') as f:
					f.write(k+" "+v+'\n')


	def load_doc(filename):
	# open the file as read only
		file = open(filename, 'r')
		# read all text
		text = file.read()
		# close the file
		file.close()
		return text

# load a pre-defined list of photo identifiers
	def load_set(filename):
		doc = load_doc(filename)
		dataset = list()
		# process line by line
		for line in doc.split('\n'):
			# skip empty lines
			if len(line) < 1:
				continue
			# get the image identifier
			identifier = line.split('.')[0]
			dataset.append(identifier)
		return set(dataset)

# # load clean descriptions into memory
	def load_clean_descriptions(filename, dataset):
	# load document
		doc = load_doc(filename)
		descriptions = dict()
		for line in doc.split('\n'):
			# split line by white space
			tokens = line.split()
			# split id from description
			if len(tokens)>0:
				image_id, image_desc = tokens[0], tokens[1:]
				# skip images not in the set
				if image_id in dataset:
					# create list
					if image_id not in descriptions:
						descriptions[image_id] = list()
					# wrap description in tokens
					desc =  ' '.join(image_desc)
					# store
					descriptions[image_id].append(desc)
		return descriptions

# # load photo features
	def load_photo_features(filename, dataset):
	# load all features
		all_features = load(open(filename, 'rb'))
		# filter features
		features = {k: all_features[k] for k in dataset}
		return features

# # covert a dictionary of clean descriptions to a list of descriptions
	def to_lines(descriptions):
		all_desc = list()
		for key in descriptions.keys():
			[all_desc.append(d) for d in descriptions[key]]
		return all_desc

# # fit a tokenizer given caption descriptions
	def create_tokenizer(descriptions):
		lines = to_lines(descriptions)
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(lines)
		return tokenizer
	def define_model(vocab_size, max_length):
	# feature extractor model
		inputs1 = Input(shape=(4096,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)
		# sequence model
		inputs2 = Input(shape=(max_length,))
		se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)
		se3 = LSTM(256)(se2)
		# decoder model
		decoder1 = add([fe2, se3])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(vocab_size, activation='softmax')(decoder2)
		# tie it together [image, seq] [word]
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# summarize model
		print(model.summary())
		
		return model
# # create sequences of images, input sequences and output words for an image
	def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):	
		X1, X2, y = list(), list(), list()
		# walk through each image identifier
		for key, desc_list in descriptions.items():
			# walk through each description for the image
			for desc in desc_list:
				# encode the sequence
				seq = tokenizer.texts_to_sequences([desc])[0]
				# split one sequence into multiple X,y pairs
				for i in range(1, len(seq)):
					# split into input and output pair
					in_seq, out_seq = seq[:i], seq[i]
					# pad input sequence
					in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
					# encode output sequence
					out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
					# store
					X1.append(photos[key][0])
					X2.append(in_seq)
					y.append(out_seq)
		return array(X1), array(X2), array(y)

	update_changes(os.path.join(app.config['UPLOAD_FOLDER'], 'description_r.txt'),os.path.join(app.config['UPLOAD_FOLDER'], 'descriptions_fb.txt'),True)
	update_changes(os.path.join(app.config['UPLOAD_FOLDER'], 'train_r.txt'),os.path.join(app.config['UPLOAD_FOLDER'], 'train.txt'),False)
	
	print("-------------DONE---------------")
    
	filename = os.path.join('train.txt')
	train = load_set(filename)
	print('Dataset: %d' % len(train))
	# descriptions
	train_descriptions = load_clean_descriptions('descriptions_fb.txt', train)
	print('Descriptions: train=%d' % len(train_descriptions))
	# photo features
	train_features = load_photo_features( 'features.pkl', train)
	print('Photos: train=%d' % len(train_features))
	# prepare tokenizer
	tokenizer = create_tokenizer(train_descriptions)
	vocab_size = len(tokenizer.word_index) + 1
	print('Vocabulary Size: %d' % vocab_size)
	# determine the maximum sequence length
	max_length = 114
	print('Description Length: %d' % max_length)
	# prepare sequences
	filename = os.path.join('test.txt')
	test = load_set(filename)
	print('Dataset: %d' % len(test))
# descriptions
	test_descriptions = load_clean_descriptions('descriptions_fb.txt', test)
	print('Descriptions: test=%d' % len(test_descriptions))
# photo features
	test_features = load_photo_features('features.pkl', test)
	print('Photos: test=%d' % len(test_features))
# prepare sequences
	X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

	
	X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
	#model = load_model(os.path.join(app.config['MODEL'], 'model-ep001-loss1.290-val_loss1.011.h5'))
	model = define_model(vocab_size, max_length)
# define checkpoint callback
	filepath = os.path.join(app.config['UPLOAD_FOLDER'],'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
	model.fit([X1train, X2train], ytrain,epochs=1, verbose=2, validation_data=([X1test, X2test], ytest))
	model.save('model_.h5')
	return render_template('index.html')

@app.route('/recent',methods=['POST'])
def recent():
	img_url_base = "http://images.nypl.org/index.php?id="
	url_for_FSA = 'http://api.repo.nypl.org/api/v1/items/e5462600-c5d9-012f-a6a3-58d385a7bc34?withTitles=yes&page={0}&per_page={1}'
	token = 'gy5zj18sf99ddtxn'
	deriv_type = 'r'.lower()
	filename = os.path.join(app.config['UPLOAD_FOLDER'],'dateDigitized.txt')
	foldername = 'Downloads_2'
	current = 1
	total_pages = 1
	count = 500

	if not os.path.isfile(filename):
		print("File not found!")
		return
	else:
		read_rec = open(filename, 'r')
		rec_date = read_rec.read()
		rec_date = datetime.datetime.strptime(rec_date,"%Y-%m-%dT%H:%M:%S")
		recent_date = rec_date
		read_rec.close()
	    
	while current <= total_pages:
		url = url_for_FSA.format(current, count)
		print(url)
		current += 1
		response_for_FSA = requests.get(url, headers={'Authorization ':'Token token=' + token}).json()
		result_list = response_for_FSA['nyplAPI']['response']
		total_pages = int(response_for_FSA['nyplAPI']['request']['totalPages'])
		for i in range(len(result_list['capture'])):
			if result_list['capture'][i]['typeOfResource'] == 'still image':               
              
                # dateDigitized recording the recent most uuid apdated with the ALT Text
				try:
					curr_date = datetime.datetime.strptime(str(result_list['capture'][i]['dateDigitized']),"%Y-%m-%dT%H:%M:%SZ")
        
					print(rec_date)
					if rec_date < curr_date:
					
						if recent_date < curr_date:
							recent_date = curr_date
							rec = open(filename, 'w')
							rec.write(str(recent_date.isoformat()))
							print("dateDigitized", recent_date.isoformat())
							rec.close()
                    
						capture_id = str(result_list['capture'][i]['imageID'])

						if not os.path.isfile(foldername + '/' + capture_id + deriv_type + '.jpg'):
							try:
								urllib.request.urlretrieve(img_url_base + capture_id + '&t='+deriv_type,foldername + '/' + capture_id + deriv_type + '.jpg')
								model = load_model(os.path.join(app.config['MODEL'], 'model-ep001-loss1.292-val_loss0.992.h5'))
								max_length = 16
								img_path=foldername + '/' + capture_id + deriv_type + '.jpg'
								print(img_path)
								photo = extract_features(img_path)
								img = Image.open(img_path)
								tokenizer=load(open((os.path.join(app.config['MODEL'],'tokenizer_fb.pkl')),"rb"))
								description = generate_desc(model, tokenizer, photo, max_length)
								
								with open(os.path.join(app.config['UPLOAD_FOLDER'], 'description_r.txt'), 'a+') as f:
									f.write(str(capture_id)+" "+str(description)+'\n')
							except Exception as e:
								print(e)
								continue
				except Exception as e:
					print(e)
					continue
	return render_template('index.html')              

@app.route('/getMetrics',methods=['POST'])
def getMetrics():
	def load_doc(filename):
	# open the file as read only
		file = open(filename, 'r')
		# read all text
		text = file.read()
		# close the file
		file.close()
		return text

# load a pre-defined list of photo identifiers
	def load_set(filename):
		doc = load_doc(filename)
		dataset = list()
		# process line by line
		for line in doc.split('\n'):
			# skip empty lines
			if len(line) < 1:
				continue
			# get the image identifier
			identifier = line.split('.')[0]
			dataset.append(identifier)
		return set(dataset)

# load clean descriptions into memory
	def load_clean_descriptions(filename, dataset):
		# load document
	    
		doc = load_doc(filename)
		descriptions = dict()
		for line in doc.split('\n'):
			# split line by white space
			tokens = line.split()
			if len(tokens)>0:
			# split id from description
				image_id, image_desc = tokens[0], tokens[1:]
				# skip images not in the set
				if image_id in dataset:
					# create list
					if image_id not in descriptions:
						descriptions[image_id] = list()
					# wrap description in tokens
					desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
					# store
					descriptions[image_id].append(desc)
		return descriptions

# load photo features
	def load_photo_features(filename, dataset):
		# load all features
		all_features = load(open(filename, 'rb'))
		# filter features
		features = {k: all_features[k] for k in dataset}
		return features

# covert a dictionary of clean descriptions to a list of descriptions
	def to_lines(descriptions):
		all_desc = list()
		for key in descriptions.keys():
			[all_desc.append(d) for d in descriptions[key]]
		return all_desc

# fit a tokenizer given caption descriptions
	def create_tokenizer(descriptions):
		lines = to_lines(descriptions)
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(lines)
		return tokenizer

# calculate the length of the description with the most words
	def max_length(descriptions):
		lines = to_lines(descriptions)
		return max(len(d.split()) for d in lines)

# map an integer to a word
	def word_for_id(integer, tokenizer):
		for word, index in tokenizer.word_index.items():
			if index == integer:
				return word
		return None

# generate a description for an image
	def generate_desc(model, tokenizer, photo, max_length):
		# seed the generation process
		in_text = 'startseq'
		# iterate over the whole length of the sequence
		for i in range(max_length):
			# integer encode input sequence
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			# pad input
			sequence = pad_sequences([sequence], maxlen=max_length)
			# predict next word
			yhat = model.predict([photo,sequence], verbose=0)
			# convert probability to integer
			yhat = argmax(yhat)
			# map integer to word
			word = word_for_id(yhat, tokenizer)
			# stop if we cannot map the word
			if word is None:
				break
			# append as input for generating the next word
			in_text += ' ' + word
			# stop if we predict the end of the sequence
			if word == 'endseq':
				break
		return in_text

# evaluate the skill of the model
	def evaluate_model(model, descriptions, photos, tokenizer, max_length):
		actual, predicted = list(), list()
		# step over the whole set
		for key, desc_list in descriptions.items():
			# generate description
			yhat = generate_desc(model, tokenizer, photos[key], max_length)
			# store actual and predicted
			references = [d.split() for d in desc_list]
			actual.append(references)
			predicted.append(yhat.split())
		# calculate BLEU score
	
		bl1=corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
		bl2=corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
		bl3=corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
		bl4=corpus_bleu(actual, predicted,  weights=(0.25, 0.25, 0.25, 0.25))
		print('BLEU-1: %f' % bl1)
		print('BLEU-2: %f' % bl2)
		print('BLEU-3: %f' % bl3)
		print('BLEU-4: %f' % bl4)
		op='BLEU-1: %f' % bl1+ '\n' +  'BLEU-2: %f' % bl2 + '\n'+ 'BLEU-3: %f' % bl3 +'\n' + 'BLEU-4: %f' % bl4
		
		return op
		
		

# prepare tokenizer on train set

# load training dataset (6K)
	filename =  'train.txt'
	dec='descriptions_fb.txt'
	train = load_set(filename)
	print('Dataset: %d' % len(train))
	# descriptions
	train_descriptions = load_clean_descriptions(dec, train)
	print('Descriptions: train=%d' % len(train_descriptions))
	# prepare tokenizer
	tokenizer = create_tokenizer(train_descriptions)
	vocab_size = len(tokenizer.word_index) + 1
	print('Vocabulary Size: %d' % vocab_size)
	# determine the maximum sequence length
	max_length = 16
	print('Description Length: %d' % max_length)

	# prepare test set

	# load test set
	filename = 'test.txt'
	test = load_set(filename)
	print('Dataset: %d' % len(test))
	# descriptions
	test_descriptions = load_clean_descriptions(dec, test)
	print('Descriptions: test=%d' % len(test_descriptions))
	# photo features
	test_features = load_photo_features('features.pkl', test)
	print('Photos: test=%d' % len(test_features))

	# load the model


	filename = os.path.join(app.config['MODEL'], 'model-ep001-loss1.292-val_loss0.992.h5')
	model = load_model(filename)
	# evaluate model
	score=evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
	resp = jsonify({'message' : score})
	resp.status_code = 201
	return resp


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    	app.run(port=3000)
	
	