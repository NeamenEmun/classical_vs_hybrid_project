## classical_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

class SimpleNN(nn.Module):  ##basic 3-layer MLP architecture from PyTorch documentation; creates a neural network that compresses input features through 2 hidden layers to output 2 classes
    def __init__(self, inputDim, numClasses):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputDim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, numClasses)
        )
    def forward(self, x):
        return self.net(x)

def trainClassical(xTrain, yTrain, xVal, yVal, inputDim, numClasses, epochs=100, batchSize=32, lr=1e-3, device='cpu'):  ##standard PyTorch training pattern from tutorials; loops through epochs, calculates loss, backpropagates gradients, and updates model weights
    model = SimpleNN(inputDim, numClasses).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainDs = TensorDataset(torch.FloatTensor(xTrain), torch.LongTensor(yTrain))
    trainLoader = DataLoader(trainDs, batch_size=batchSize, shuffle=True)
    start = time.time()
    bestAcc = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in trainLoader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    
    trainTime = time.time() - start
    return model, trainTime
