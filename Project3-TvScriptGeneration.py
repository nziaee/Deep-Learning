#Create batch data
def batch_data(words, sequence_length, batch_size):
    batch_size_total = batch_size * sequence_length
    num_batches = len(words)//batch_size_total
    words = words[:num_batches*batch_size_total]
    #words = words.reshape((batch_size, -1))
    #print(words.shape[1])
    features = []
    targets = []
    for n in range(0, len(words)-1):
        if(n+sequence_length <= len(words)):
            x = words[n:n+sequence_length]
            features.append(x)
            try:
                targets.append(words[n+sequence_length])
            except IndexError:
                targets.append(words[0])       
    feature_tensors = torch.Tensor(features)
    target_tensors = torch.Tensor(targets)
    data = TensorDataset(feature_tensors, target_tensors)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True) #shuffle=True
	
    return data_loader

#Create RNN model	
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.5):
		self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_size = output_size
		#I was having error based on some students(@Reham El-Kholy) in slack who were having similar issue adding the embedding solved the issue. 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)
		
	def forward(self, nn_input, hidden):
		embedding = self.embedding(nn_input.long())
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
		
		#I was having trouble passing the test because of output size I used the (@Reham El-Kholy) example code to 
        #figure out how to reshape and get the output
        #print(output.size())
        batch_size = nn_input.size(0)
        output = output.reshape(batch_size,-1,self.output_size)
        #print(output.size())
        output = output[:,-1]
        #print(output.size())
        # return one batch of output word scores and the hidden state
        return output, hidden
		
def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
        # initialize hidden state with zero weights, and move to GPU if available
        if(train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                     weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden
		
#Define forward and backpropagation		
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
	# perform backpropagation and optimization
    hidden = tuple([each.data for each in hidden])
    rnn.zero_grad()
    output, hidden = rnn(inp, hidden)
    loss = criterion(output.squeeze(), target.long())
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(), 10)
    optimizer.step()
    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden
	
#Data params
# Sequence Length
sequence_length = 8  # of words in a sequence
# Batch Size
batch_size = 300

# Training parameters
# Number of Epochs
num_epochs = 20
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = vocab_size
# Embedding Dimension
embedding_dim = 300
# Hidden Dimension
hidden_dim = 350
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 20