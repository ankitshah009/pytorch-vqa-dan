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
    def __init__(self, textencoder, visual_hidden_size, text_hidden_size, answer_size, k=2, use_cuda=False):
        super(rDAN, self).__init__()

        hidden_size = textencoder.bilstm.hidden_size
        memory_size = 2 * hidden_size
        
        # Visual
        #self.visualencoder = visualencoder
        self.Wv = nn.Linear(in_features=2048, out_features=visual_hidden_size)
        self.Wvm = nn.Linear(in_features=memory_size, out_features=visual_hidden_size)
        self.Wvh = nn.Linear(in_features=visual_hidden_size, out_features=1)
        self.P = nn.Linear(in_features=2048, out_features=memory_size)
    
        # Text
        self.textencoder = textencoder
        self.Wu = nn.Linear(in_features=2*hidden_size, out_features=text_hidden_size)
        self.Wum = nn.Linear(in_features=memory_size, out_features=text_hidden_size)
        self.Wuh = nn.Linear(in_features=text_hidden_size, out_features=1)
 
        self.Wans = nn.Linear(in_features=memory_size, out_features=answer_size)
        
        self.classifier = Classifier(memory_size, hidden_size, answer_size, 0.5)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Activations
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0) # Softmax over first dimension

        # Loops
        self.k = k

        self.use_cuda = use_cuda

    def forward(self, feature, text, choices=None, max_length=50):
        # Convert input into representations
        # Visual (nregion, batch_size, dim)
        batch_size = feature.shape[0]
        feature = feature.view(batch_size, 2048, -1)

        #visual_output = self.visualencoder.forward(image) # ( batch_size, dim, nregion)
        #vns = visual_output.permute(2,0,1) # (nregion, batch_size, dim)
        vns = feature.permute(2,0,1) # (nregion, batch_size, dim)

        # Text Inputs are of shape (seq_len, batch_size)
        text = text.permute(1,0)[:max_length]
        uts = self.textencoder.forward(text) # (seq_len, batch_size, dim)

        u = uts.mean(0)
        v = self.tanh( self.P( vns.mean(0) ))

        memory = v * u

        seq_len = uts.shape[0]
        # This is the main loop of the program
        # K indicates the hop
        for k in range(self.k):
            # This iterate over the number of word tokens
            # Visual
            hv = self.tanh(self.Wv(self.dropout(vns))) * self.tanh(self.Wvm(self.dropout(memory)))

            # (seq_len, batch_size, memory_size)
            # attention weights for every region
            alphaV = self.softmax(self.Wvh(self.dropout(hv)))

            v = self.tanh(self.P(alphaV * vns)).sum(0)
            # Text 
            # (seq_len, batch_size, dim) * (batch_size, dim)
            hu = self.tanh(self.Wu(self.dropout(uts))) * self.tanh(self.Wum(self.dropout(memory)))

            # (seq_len, batch_size, memory_size)
            alphaU = self.softmax(self.Wuh(self.dropout(hu)))
            
            u = (alphaU * uts).sum(0) # Sum over sequence
            
            # Build Memory
            memory = memory + u * v
        
        # We compute scores using a classifier
        scores = self.classifier(memory)

        return scores

if __name__ == "__main__":
    pass
