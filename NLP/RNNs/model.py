import torch

class RNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        output, hidden = self.rnn(embedded)
        
        assert torch.equal(output[-1,:,:], hidden.squeeze(0))
        
        out = self.linear(hidden)
        return out

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, hidden_dim, static=False, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.dropout = torch.nn.Dropout(p=dropout)

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        if static:
            self.embedding.weight.requires_grad = False

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, 
                                         num_layers=num_layers,
                                         bidirectional=True, 
                                         dropout=dropout, 
                                         batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim*num_layers*2, 1)
    
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = torch.transpose(embedded, dim0=1, dim1=0)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.linear(self.dropout(torch.cat([cell[i,:, :] for i in range(cell.shape[0])], dim=1)))
        return out
