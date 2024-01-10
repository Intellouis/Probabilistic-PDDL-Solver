import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = self.fc2(output)
        return output
    
    def train(self, input, target):
        self.optimizer.zero_grad()
        output = self(input)
        target = torch.autograd.Variable(target, requires_grad=False)
        # loss = F.mse_loss(output, target)
        # loss = F.binary_cross_entropy_with_logits(output, target)
        # loss = torchvision.ops.sigmoid_focal_loss(output, target, reduction="sum")
        loss = F.binary_cross_entropy_with_logits(output, target)           # I'm not sure about this loss function      tensor(0.7052)  ('loss.size()' is ())
        # loss = torch.autograd.Variable(loss, requires_grad=True)
        # loss.require_grad_(True)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# class MLP_sigmoid(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
        
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
        
#     def forward(self, input):
#         output = F.relu(self.fc1(input))
#         output = self.fc2(output)
#         return F.sigmoid(output)
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.model_1 = MLP(input_size, hidden_size, output_size)
        self.model_2 = MLP(input_size, hidden_size, output_size)
        self.model_3 = MLP(input_size, hidden_size, output_size)
        self.model_4 = MLP(input_size, hidden_size, output_size)
        self.model_5 = MLP(input_size, hidden_size, output_size)
        self.model_6 = MLP(input_size, hidden_size, output_size)
        self.model_7 = MLP(input_size, hidden_size, output_size)
        self.model_8 = MLP(input_size, hidden_size, output_size)
        # self.model_on = MLP_sigmoid(2 * output_size, hidden_size, 1)
        # self.model_clear = MLP_sigmoid(output_size, hidden_size, 1)
        self.model_on = MLP(2 * output_size, hidden_size, 1)
        self.model_clear = MLP(output_size, hidden_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, input):
        # print("input.size()", input.size())         # (1, 24)
        block_embedding = []
        block_embedding.append(self.model_1(input))
        block_embedding.append(self.model_2(input))
        block_embedding.append(self.model_3(input))
        block_embedding.append(self.model_4(input))
        block_embedding.append(self.model_5(input))
        block_embedding.append(self.model_6(input))
        block_embedding.append(self.model_7(input))
        block_embedding.append(self.model_8(input)) 
        # print("block_embedding[-1].size()", block_embedding[-1].size())         # (1, 64)
        # predicate: on & clear
        output = []
        # output = torch.Tensor(64)
        idx = -1
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                idx += 1
                output.append(self.model_on(torch.cat((block_embedding[i], block_embedding[j]), dim=1)))#.squeeze(0))
                # output[idx] = self.model_on(torch.cat((block_embedding[i], block_embedding[j]), dim=1)).squeeze(0)
        for i in range(8):
            output.append(self.model_clear(block_embedding[i]))#.squeeze(0))
            # output[56+i] = self.model_clear(block_embedding[i]).squeeze(0)
        # print("output[-1].size()", output[-1].size())            # (1,)
        # output = torch.Tensor(output)
        output = torch.cat(output, dim=1)
        # print("output.size()", output.size())                    # (64,)
        return output




    def train(self, input, target):
        self.optimizer.zero_grad()
        output = self(input)
        target = torch.autograd.Variable(target, requires_grad=False)
        # loss = F.mse_loss(output, target)
        # loss = F.binary_cross_entropy_with_logits(output, target)
        # loss = torchvision.ops.sigmoid_focal_loss(output, target, reduction="sum")
        loss = F.binary_cross_entropy_with_logits(output, target)           # I'm not sure about this loss function      tensor(0.7052)  ('loss.size()' is ())
        # loss = torch.autograd.Variable(loss, requires_grad=True)
        # loss.require_grad_(True)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def test(self, input, target):
        output = self(input)
        loss = F.binary_cross_entropy_with_logits(output, target)
        return loss.item()
    
    def check_parameters_value(self):
        lst = []
        for name, param in self.named_parameters():
            # print(name, param.data)
            lst.append(param.data)
        return lst


if __name__ == '__main__':
    encoder = Encoder(1, 2, 3)
    # print(encoder.parameters())
    for name, param in encoder.named_parameters():
        print(name, param.size())