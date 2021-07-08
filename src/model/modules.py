import torch
import torch.nn as nn
import math

class DenseBlock(nn.Module):
    def __init__(self, dilation: int, in_filters: int, out_filters: int):
        super().__init__()
        self.dilatation = dilation
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.causal_conv1 = nn.Sequential(
            nn.ConstantPad1d((dilation, 0), 0),
            nn.Conv1d(in_filters, out_filters, kernel_size=2,
                                   dilation=dilation)
        )
        self.causal_conv2 = nn.Sequential(
            nn.ConstantPad1d((dilation, 0), 0),
            nn.Conv1d(in_filters, out_filters, kernel_size=2, dilation=dilation)
        )        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input, output = (time x filters)
        input = input.unsqueeze(2)
        xf, xg = self.causal_conv1(input), self.causal_conv2(input)
        activations = self.tanh(xf) * self.sigmoid(xg)
        return torch.cat([input, activations], dim=1).squeeze(2)

class TCBlock(nn.Module):
    def __init__(self, seq_len: int, in_filters: int, filters: int):
        super().__init__()
        self.no_layer = int(math.ceil(math.log2(seq_len)))
        self.model = nn.Sequential()
        for i in range(self.no_layer):
            block = DenseBlock(2 ** i, (in_filters + i * filters), filters)
            self.model.add_module(f'dense_{i + 1}', block)

    def forward(self, input):
        output = self.model(input)
        return output

class AttentionBlock(nn.Module):
    def __init__(self, input_size, key_size, value_size):
        super().__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.key_layer = nn.Linear(input_size, key_size)
        self.query_layer = nn.Linear(input_size, key_size)
        self.value_layer = nn.Linear(input_size, value_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
       # input = (time x input_size)
       # output = (time x (input_size + value_size)
        seq_length = input.shape[0]
        keys = self.key_layer(input)  # bs x t x ks
        query = self.query_layer(input)  # bs x t x ks
        dot_product = query@keys.T
        mask = torch.ones(seq_length, seq_length, dtype=torch.bool) 
        for i in range(seq_length):
            mask[i, i:] = False
        dot_product[mask] = - float('inf')
        probs = self.softmax(dot_product / math.sqrt(self.key_size))
        values = self.value_layer(input)
        read = probs.matmul(values)
        output = torch.cat([input, read], dim=-1)
        return output
