import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext.vocab as vocab

from model import Classifier

class TextEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, 
                                batch_first=False, 
                                bidirectional=True,
                                dropout=0.5)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        embed_output = self.embed(x)
        bilstm_output, _ = self.bilstm(self.dropout(embed_output))
        return bilstm_output

    def load_pretrained(self, dictionary):
        print("Loading pretrained weights...")
        # Load pretrained vectors for embedding layer
        glove = vocab.GloVe(name='6B', dim=self.embed.embedding_dim)

        # Build weight matrix here
        pretrained_weight = self.embed.weight.data
        for word, idx in dictionary.items():
            if word.lower() in glove.stoi:     
                vector = glove.vectors[ glove.stoi[word.lower()] ]
                pretrained_weight[ idx ] = vector

        self.embed.weight = nn.Parameter(pretrained_weight)       

class rDAN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, answer_size, k=2):
        super(rDAN, self).__init__()

        # Build Text Encoder
        self.textencoder = TextEncoder(num_embeddings=num_embeddings, 
                                       embedding_dim=embedding_dim, 
                                        hidden_size=hidden_size)

        memory_size = 2 * hidden_size # bidirectional
        
        # Visual Attention
        self.Wv = nn.Linear(in_features=2048, out_features=hidden_size)
        self.Wvm = nn.Linear(in_features=memory_size, out_features=hidden_size)
        self.Wvh = nn.Linear(in_features=hidden_size, out_features=1)
        self.P = nn.Linear(in_features=2048, out_features=memory_size)
    
        # Textual Attention
        self.Wu = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.Wum = nn.Linear(in_features=memory_size, out_features=hidden_size)
        self.Wuh = nn.Linear(in_features=hidden_size, out_features=1)
 
        self.Wans = nn.Linear(in_features=memory_size, out_features=answer_size)
        
        # Scoring Network
        self.classifier = Classifier(memory_size, hidden_size, answer_size, 0.5)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0) # Softmax over first dimension

        # Loops
        self.k = k

    def forward(self, visual, text):

        batch_size = visual.shape[0]

        # Prepare Visual Features
        visual = visual.view(batch_size, 2048, -1)
        vns = visual.permute(2,0,1) # (nregion, batch_size, dim)

        # Prepare Textual Features
        text = text.permute(1,0)
        uts = self.textencoder.forward(text) # (seq_len, batch_size, dim)

        # Initialize Memory
        u = uts.mean(0)
        v = self.tanh( self.P( vns.mean(0) ))
        memory = v * u

        # K indicates the number of hops
        for k in range(self.k):
            # Compute Visual Attention
            hv = self.tanh(self.Wv(self.dropout(vns))) * self.tanh(self.Wvm(self.dropout(memory)))
            # attention weights for every region
            alphaV = self.softmax(self.Wvh(self.dropout(hv))) #(seq_len, batch_size, memory_size)
            # Sum over regions
            v = self.tanh(self.P(alphaV * vns)).sum(0)

            # Text 
            # (seq_len, batch_size, dim) * (batch_size, dim)
            hu = self.tanh(self.Wu(self.dropout(uts))) * self.tanh(self.Wum(self.dropout(memory)))
            # attention weights for text features
            alphaU = self.softmax(self.Wuh(self.dropout(hu)))  # (seq_len, batch_size, memory_size)
            # Sum over sequence
            u = (alphaU * uts).sum(0) # Sum over sequence
            
            # Build Memory
            memory = memory + u * v
        
        # We compute scores using a classifier
        scores = self.classifier(memory)

        return scores

if __name__ == "__main__":
    pass
