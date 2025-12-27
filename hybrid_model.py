##hybrid_model.py
import pennylane as qml
from pennylane import numpy as pnp
import torch
from torch import nn, optim
import time

nQubits = 2
dev = qml.device("default.qubit", wires=nQubits)

def quantumCircuit(inputs, weights):  ##lines 10-17 is from pennylane documentation; encodes input data as rotations on qubits, applies learnable weight rotations, entangles qubits, and measures result
    for i in range(nQubits):
        qml.RY(inputs[i], wires=i)
    for i in range(nQubits):
        qml.RX(weights[i], wires=i)
    for i in range(nQubits - 1):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0))

qnode = qml.QNode(quantumCircuit, dev, interface="torch", diff_method="backprop")

class QuantumLayer(nn.Module):
    def __init__(self, qnode, nQubits, hiddenDim):
        super().__init__()
        self.qnode = qnode
        self.nQubits = nQubits
        self.weights = nn.Parameter(torch.randn(nQubits) * 0.1)
    
    def forward(self, x):  ##batch processing adapted from PennyLane documentation; processes each sample individually through quantum circuit and stacks results into batch
        batchSize = x.shape[0]
        results = []
        
        for i in range(batchSize):
            result = self.qnode(x[i], self.weights)
            results.append(result)
        
        output = torch.stack(results)
        output = output.float()
        return output.unsqueeze(1)

class HybridNet(nn.Module):  ##hybrid quantum-classical architecture design from PennyLane tutorials; compresses input features, sends to quantum circuit, processes quantum output through final layer
    def __init__(self, inputDim, nQubits, numClasses):
        super().__init__()
        hiddenDim = max(64, inputDim // 32)
        self.fc1 = nn.Linear(inputDim, hiddenDim)
        self.fc1Relu = nn.ReLU()
        self.fc1Bn = nn.BatchNorm1d(hiddenDim)
        
        self.fcToQubits = nn.Linear(hiddenDim, nQubits)
        self.quantumLayer = QuantumLayer(qnode, nQubits, nQubits)
        self.fcPost = nn.Linear(1, 8)  ##post-quantum hidden layer for fair capacity vs classical
        self.fcPostRelu = nn.ReLU()
        self.fc2 = nn.Linear(8, numClasses)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1Relu(x)
        x = self.fc1Bn(x)
        x = torch.tanh(self.fcToQubits(x))
        x = self.quantumLayer(x)
        x = self.fcPost(x)
        x = self.fcPostRelu(x)
        x = self.fc2(x)
        return x

def trainHybrid(xTrain, yTrain, xTest, yTest,
                inputDim, numClasses, nQubits=2,
                epochs=30, batchSize=16, lr=0.001, device="cpu"):

    model = HybridNet(inputDim, nQubits, numClasses).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    xTrainT = torch.FloatTensor(xTrain).to(device)
    yTrainT = torch.LongTensor(yTrain).to(device)
    xTestT  = torch.FloatTensor(xTest).to(device)
    yTestT  = torch.LongTensor(yTest).to(device)

    trainDs = torch.utils.data.TensorDataset(xTrainT, yTrainT)
    trainLoader = torch.utils.data.DataLoader(trainDs, batch_size=batchSize, shuffle=True)

    start = time.time()
    losses = []

    for epoch in range(epochs):
        model.train()
        epochLoss = 0.0
        batchCount = 0
        
        for xb, yb in trainLoader:  ##lines 86-93 is from pytorch tutorial; standard training loop that predicts, calculates cross-entropy loss, backpropagates, and updates parameters
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
            batchCount += 1
        
        avgLoss = epochLoss / batchCount if batchCount > 0 else 0
        losses.append(avgLoss)
        print(f"Hybrid Epoch {epoch + 1}/{epochs}, Loss: {avgLoss:.4f}")

    trainTime = time.time() - start

    model.eval()
    with torch.no_grad():
        preds = model(xTestT)
        correct = (preds.argmax(axis=1) == yTestT).sum().item()
        accuracy = correct / len(yTest)

    return model, accuracy, trainTime, losses
