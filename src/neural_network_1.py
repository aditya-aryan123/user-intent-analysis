import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data into a pandas DataFrame
df = pd.read_csv("../input/online_shoppers_intention (1).csv")

# Prepare the input data by converting categorical variables into one-hot encodings
df = pd.get_dummies(df,
                    columns=["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"])

# Scale the input data using StandardScaler
scaler = StandardScaler()
df[["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated",
    "ProductRelated_Duration", "BounceRates in %", "ExitRates in %", "PageValues", "SpecialDay (probability)"]] = \
    scaler.fit_transform(df[[
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated",
        "ProductRelated_Duration", "BounceRates in %", "ExitRates in %", "PageValues", "SpecialDay (probability)"]])

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Convert the data into tensors
train_inputs = torch.tensor(train_df.drop("Revenue", axis=1).values, dtype=torch.float32)
train_labels = torch.tensor(train_df["Revenue"].values, dtype=torch.float32).reshape(-1, 1)
test_inputs = torch.tensor(test_df.drop("Revenue", axis=1).values, dtype=torch.float32)
test_labels = torch.tensor(test_df["Revenue"].values, dtype=torch.float32).reshape(-1, 1)


# Create a neural network model using PyTorch
class RevenueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(df.shape[1] - 1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


model = RevenueModel()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


# Define a function for calculating accuracy
def accuracy(pred, labels):
    return (pred.round().long() == labels.long()).float().mean().item()


# Train the model
num_epochs = 100
best_loss = float("inf")
early_stopping_counter = 0
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(train_inputs)
    loss = criterion(predictions, train_labels)
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if the loss has improved
    if loss.item() < best_loss:
        best_loss = loss.item()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= 10:
            break

    # Print the accuracy
    train_acc = accuracy(predictions, train_labels)
    test_predictions = model(test_inputs)
    test_acc = accuracy(test_predictions, test_labels)
    print("Epoch:", epoch)
    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)
    print("Loss:", loss.item())

    train_acc = accuracy(predictions, train_labels)
    test_predictions = model(test_inputs)
    test_acc = accuracy(test_predictions, test_labels)
    print("Epoch:", epoch)
    print("Train accuracy:", train_acc)
    print("Test accuracy:", test_acc)
    print("Loss:", loss.item())

    test_predictions = model(test_inputs)
    test_acc = accuracy(test_predictions, test_labels)
    print("Final test accuracy:", test_acc)
