from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tarfile
import os
import pandas as pd
import re
import nltk
import numpy as np

class Data():
	#TODO create path using os function
	def __init__(self, dim_size=50, max_len=500, max_len_target=35, frac=1):
		self.dim_size = dim_size
		self.embedding_path = './data/glove.6B.50d.txt.tar.gz'
		self.txt_path = './data/glove.6B.{}d.txt'.format(dim_size)
		self.tar_path = './data/cnn.tgz'
		self.file_path = './data/cnn/stories/'
		self.clean_data_path = './data/clean_data.csv'
		self.story_list = []
		self.highlights_list = []
		self.dummy_highlight_list = []
		self.max_len = max_len
		self.max_len_target = max_len_target
		
		self.file_list = os.listdir(self.file_path)
		self.file_list = self.file_list[:int(frac*len(self.file_list))]
		print(len(self.file_list))

	def load_data(self):
		#extract file
		if os.path.isfile(self.tar_path) and os.path.isdir(self.file_path) == False:
			tar = tarfile.open(self.tar_path, 'r')
			tar.extractall('./data/')
			tar.close

		for filename in self.file_list:
			with open(self.file_path+filename, 'r') as f:
				text = ''
				for line in f:
					text += line.lower()

				story, highlights = self.split_data(text)
				self.story_list.append(story)
				self.highlights_list.append(highlights)
				f.close()

	def split_data(self, text):
		index = text.find('@highlight')
		story = text[:index]
		highlights = text[index:].split('@highlight')
		highlights = [h.strip() for h in highlights if len(h) > 0]

		return story, highlights

	#save dummy data
	def preprocess_data(self, save=False):
		if len(self.story_list) == 0:
			self.load_data()	

		for i, story in enumerate(self.story_list):
			index = story.find('(cnn) -- ')
			if index > -1:
				self.story_list[i] = story[index+len('(cnn) -- '):]

		#add <START> and <EOS> token to indicate start and end of sentence
		for i, highlights in enumerate(self.highlights_list):
			for j, highlight in enumerate(highlights):
				self.highlights_list[i][j] = '<START> ' + self.highlights_list[i][j] + ' <EOS>'
				
				#create dummy highlight list, maybe not the best practice here
				#this list is used to fit on text
				self.dummy_highlight_list.append('<START> ' + self.highlights_list[i][j] + ' <EOS>')


	def get_data(self, verify=False, first_highlight_only=True):
		self.preprocess_data()

		input_token = Tokenizer()
		target_token = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', lower=False)

		input_token.fit_on_texts(self.story_list)
		self.input_word_index = input_token.word_index
		
		target_token.fit_on_texts(self.dummy_highlight_list)
		self.target_word_index = target_token.word_index
		#encode story
		self.story_list = input_token.texts_to_sequences(self.story_list)
		self.story_list = pad_sequences(self.story_list, maxlen=self.max_len,\
										padding='post', truncating='post')

		#encode highlight
		for i, highlights in enumerate(self.highlights_list):
				self.highlights_list[i] = target_token.texts_to_sequences(self.highlights_list[i])
				self.highlights_list[i] = pad_sequences(self.highlights_list[i],\
												maxlen=self.max_len_target, 
												padding='post', truncating='post')
		if first_highlight_only:
			for i, highlights in enumerate(self.highlights_list):
				self.highlights_list[i] = highlights[0]

		if verify:
			#this code is to verify that the encoded texts are correct
			index = 0 #set index here
			res = dict((v,k) for k,v in self.target_word_index.items())
			for num in self.highlights_list[index][0]:
				print(res.get(num), end=' ')

		#Turn highlight list into one hot
		#(None, sequence_length, target_size)
		mat_shape = (len(self.highlights_list), self.max_len_target, len(self.target_word_index))
		one_hot_matrix = np.zeros(mat_shape)
		for i, highlight in enumerate(self.highlights_list):
			for j, index in enumerate(highlight):
				one_hot_matrix[i][j][index] = 1

		self.highlights_list = one_hot_matrix
		return self.story_list, self.highlights_list

	def download_data(self):
		#if not exist download data
		pass

	def extract_embedding(self):
		#if txt file exist, extract file
		if os.path.isfile(self.txt_path) == False:
			tar = tarfile.open(self.embedding_path, 'r')
			tar.extractall('./data/')
			tar.close()

	def load_embedding(self):
		self.extract_embedding()
		
		#+1 because there is no 0 index in the embedding, it starts from 1
		#if there is no +1, the last word will not be included
		vocab_size = len(self.input_word_index) + 1

		#load embedding, put it into a dictionary
		embedding_index = dict()
		f = open(self.txt_path)
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embedding_index[word] = coefs
		f.close()

		#mapping word to the corresponding vectors
		embedding_matrix = np.zeros((vocab_size, self.dim_size))
		for word, i in self.input_word_index.items():
			embedding_vector = embedding_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector

		return embedding_matrix

	def get_input_word_index(self):
		return self.input_word_index

	def get_target_word_index(self):
		return self.target_word_index

	def get_embedding_dim(self):
		return self.dim_size