import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from encoder import MLP, Encoder, Deep_MLP


NUM_BLOCKS = 8
NUM_DIMENSION = 3
VECTOR_LENGTH = 64
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(SGN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.encoder = Encoder(input_size, hidden_size, output_size, lr)
        self.decoder = MLP(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
    
    def train(self, input, target):
        self.optimizer.zero_grad()
        output = self.forward(input)
        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def predict(self, input):
        output = self.forward(input)
        return output.detach().numpy()
    
    def evaluate(self, input, target):
        output = self.forward(input)
        loss = F.mse_loss(output, target)
        return loss.item()
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_optimizer(self):
        return self.optimizer
    
    def get_input_size(self):
        return self.input_size
    
    def get_hidden_size(self):
        return self.hidden_size
    
    def get_output_size(self):
        return self.output_size
    


if __name__ == '__main__':
    with open('./dataset/IL_dataset_2000.pkl', 'rb') as f:
        dataset = pickle.load(f)
    X = dataset["data"]
    Y = dataset["labels"]
    train_X, train_Y = X.copy(), Y.copy()
    train_X = torch.Tensor(train_X)
    train_Y = torch.Tensor(train_Y)
    encoder = Encoder(NUM_BLOCKS * 2, 128, VECTOR_LENGTH).to(device)
    
    for epoch in range(1, 51):
        for i in range(0, train_X.shape[0], BATCH_SIZE):
            x = train_X[i:i+BATCH_SIZE]
            x = x.reshape(x.shape[0], -1).to(device)
            y = train_Y[i:i+BATCH_SIZE].to(device)
            loss = encoder.train(x, y)
        if epoch % 1 == 0:
            torch.save(encoder.state_dict(), f"./model/all_2000/OSIL_state_dict_{epoch}.pth")
        # print(loss)
