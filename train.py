import models
from utils import Data
import sys
import getopt

num_epoch = 50
max_len = 500
max_len_target = 35
batch_size = 64

#This variable is for testing the code
#Set this to 1 if the network is planned to be trained
data_frac = 0.001

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:t:e:b:f:', ['input_seq=', 'target_seq=', 'epoch=', 'batch=', 'frac='])
except getopt.GetoptError:
    sys.exit(2)

for opt, arg in opts:
	print(opt, arg)
	if opt in ('-i', '--input_seq'):
		max_len = int(arg)
	elif opt in ('-t', '--target_seq'):
		max_len_target = int(arg)
	elif opt in ('-e', '--epoch'):
		num_epoch = int(arg)
	elif opt in ('-b', '--batch'):
		batch_size = int(arg)
	elif opt in ('-f', '--frac'):
		data_frac = float(arg)
	else:
		sys.exit(2)

def main():
	data = Data.Data(max_len=max_len, max_len_target=max_len_target, frac=data_frac)
	x, y = data.get_data()
	input_word_index = data.get_input_word_index()
	target_word_index = data.get_target_word_index()
	dim_size = data.get_embedding_dim()
	embedding_matrix = data.load_embedding()

	model = models.OneShotModel(x, y, num_epoch=num_epoch, max_len=max_len, \
								max_len_target=max_len_target, batch_size=batch_size, \
								input_word_index=input_word_index, \
								target_word_index=target_word_index, \
								dim_size=dim_size, embedding_matrix=embedding_matrix)

	model.train()

if __name__ == '__main__':
	main()