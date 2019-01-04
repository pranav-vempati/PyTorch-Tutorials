import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim

torch.manual_seed(42)

training_str_1 = "The dog ate the apple"

training_str_2 = "Everybody read that book"

training_data = [(training_str_1.split(), ["DET", "NN", "V", "DET", "NN"]), (training_str_2.split(), ["NN", "V", "DET", "NN"])]

word_to_index = {}

char_to_index = {}

def convert_word_to_character_index_sequence(word, char_to_ix):
	wordList = [char for char in list(word) if char != ' ']
	return torch.tensor([char_to_ix[char] for char in wordList], dtype = torch.long)


def prepare_input_sequence(word_seq,word_to_ix, char_to_ix):
	output_list = []
	for word in word_seq:
		output_list.append(torch.tensor(word_to_ix[word], dtype = torch.long), convert_word_to_character_index_sequence(word, char_to_ix))
	return output_list

def prepare_target_sequence(target_sequence, tag_to_ix):
	indices = []
	for tag in target_sequence:
		indices.append(tag_to_ix(tag))
	return torch.tensor(indices, dtype = torch.long)

	
for sentence, tags in training_data:
	for word in sentence:
		if word not in word_to_index:
			word_to_index[word] = len(word_to_index)
	for character in word:
		if character not in char_to_index and character != ' ':
			char_to_index[character] = len(char_to_index) 

tag_to_index = {"DET":0, "NN":1, "V": 2}

EMBEDDING_DIM = 12 # Maintain the same embedding dimension for words/individual characters

HIDDEN_DIM = 12

class AugmentedLSTMTagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, charset_size, tagset_size):
		super().__init__() # Python 3.x only
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.character_embeddings = nn.Embedding(charset_size, embedding_dim)
		self.character_lstm = nn.LSTM(embedding_dim, embedding_dim)
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.word_lstm = nn.LSTM(embedding_dim + embedding_dim, hidden_dim)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.char_hidden_state = self.initialize_hidden_state() # Character LSTM hidden state
		self.hidden = self.initialize_hidden_state() # This represents Word_LSTM's final hidden state. Initialize it here.

	def initialize_hidden_state(self):
		return (torch.zeros(1,1,self.hidden_dim), torch.zeros(1,1,self.hidden_dim)) # The LSTM's cell state is encapsulated in the hidden state


	def forward(self, sentence):
		concatenated_output_state = []
		word_indices = []

		for word in sentence:
			self.char_hidden_state = self.init_hidden() # Refresh hidden state, detaching it from the earlier sequence
			character_indices = word[1] # This has already been wrapped as a Torch.LongTensor
			character_level_embeddings = self.character_embeddings(character_indices) # Use the tensor to index into the lookup table
			output_lstm, self.char_hidden_state = self.character_lstm(character_level_embeddings.view(len(sentence), 1, EMBEDDING_DIM), self.char_hidden_state)
			concatenated_output_state.append(output_lstm) # Append LSTM state to the list
			word_indices.append(word[0])

		concatenated_output_state = torch.unsqueeze(concatenated_output_state,0) # Convert to a tensor, add an extra first dimension

		word_embeddings = self.word_embeddings(torch.tensor(word_indices, dtype = torch.long).view(len(sentence), 1, EMBEDDING_DIM))

		concatenated_characters_and_words = torch.cat((word_embeddings, concatenated_output_state), len(list(word_embeddings.size()))-1) # Concatenate the tensors along their last axis

		lstm_output_state, self.hidden = self.word_lstm(concatenated_characters_and_words, self.hidden)

		tag_space = self.hidden2tag(lstm_output_state.view(len(sentence),-1)) # A Linear layer mapping from tag space to scores

		tag_scores = F.log_softmax(tag_space, dim = None) # Softmax along all dimensions. Log_softmax is required for NLLLoss

		return tag_scores


	model = AugmentedLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_index), len(char_to_index), len(tag_to_index))

	criterion = nn.NLLLoss()

	optimizer = optim.SGD(model.parameters(), lr = 0.01)

	for epoch in range(100): # Training loop
		for sentence, tags in training_data:
			model.zero_grad() # Clear accumulated gradient buffers

			model.hidden = model.initialize_hidden_state()

			input_sequence = prepare_input_sequence(sentence, word_to_index, char_to_index)

			targets = prepare_target_sequence(tags, tag_to_index)

			tag_scores = model(input_sequence)

			loss = criterion(tag_scores, targets)

			loss.backward()

			optimizer.step()





















