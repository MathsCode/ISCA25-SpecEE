import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import numpy as np
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
import torch
model_name = 'Llama-7B'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = torch.load('train.pt').to(torch.float).to(device)
print(train_data.shape)
train_label = torch.load('label.pt').reshape(-1).to(train_data.device).to(train_data.dtype)
acc_list = []
for x in range(40):
    print('------------------Layer:',x,'------------------')
    train_data_layer = train_data[x::40,:]
    train_label_layer = train_label[x::40]
    dataset = TensorDataset(train_data_layer, train_label_layer)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    real_train = int(0.1 * len(train_dataset))
    other_train = len(train_dataset) - real_train
    train_dataset, others = random_split(train_dataset, [real_train, other_train])
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    model = MLP(train_data.shape[-1], 512, 1).to(device).to(torch.float)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # if (epoch+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')     
    # model.eval()
    with torch.no_grad():
        y_pred_train = []
        y_true_train = []
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X).round()
            y_pred_train.extend(outputs.cpu().numpy())
            y_true_train.extend(batch_y.cpu().numpy())
        y_pred_test = []
        y_true_test = []
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).round()
            y_pred_test.extend(outputs.cpu().numpy())
            y_true_test.extend(batch_y.cpu().numpy())

        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        test_accuracy = accuracy_score(y_true_test, y_pred_test)

        print(f'Train Accuracy: {train_accuracy:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        
    acc_list.append(test_accuracy)
    torch.save(model,'model'+str(x)+'.pth')
print(acc_list)
print(sum(acc_list)/len(acc_list))
