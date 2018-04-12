import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNLSTM, Dropout, BatchNormalization, Activation, RepeatVector, TimeDistributed

class OneShotModel():
	def __init__(self, x, y, max_len=500, max_len_target=50, batch_size=32, \
				 rnn_size=128, learning_rate=0.001, num_epoch=25, num_layer=1, 
				 input_word_index=None, dim_size=50, embedding_matrix=None, \
				 target_word_index=None):

		self.x = np.array(x)
		self.y = np.array(y)
		self.max_len = max_len
		self.max_len_target = max_len_target
		self.batch_size = batch_size
		self.rnn_size = rnn_size
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.num_layer = num_layer
		self.input_word_index = input_word_index
		self.target_word_index = target_word_index
		self.dim_size = dim_size
		self.embedding_matrix = embedding_matrix

		self.model = Sequential()
		if embedding_matrix is not None:
			#use embedding matrix
			self.model.add(Embedding(len(input_word_index)+1, dim_size, input_length=max_len, \
								weights=[embedding_matrix], trainable=False))
		else:
			#train embedding from scratch
			self.model.add(Embedding(len(input_word_index)+1, dim_size, input_length=max_len))

		#encoder
		self.model.add(CuDNNLSTM(self.rnn_size))
		self.model.add(Activation('relu'))
		self.model.add(RepeatVector(self.max_len_target))

		#decoder
		self.model.add(CuDNNLSTM(self.rnn_size, return_sequences=True))
		self.model.add(Activation('relu'))
		self.model.add(TimeDistributed(Dense(len(self.target_word_index), activation='softmax')))

		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(self.model.summary())

	def train(self):
		self.model.fit(self.x, self.y, epochs=self.num_epoch, batch_size=self.batch_size, verbose=1)
		self.model.save_weights('../one_shot_weights.h5')

