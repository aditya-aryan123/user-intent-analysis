import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def forward_pass(model, inputs):
    with torch.no_grad():
        outputs = model(inputs)
        return outputs


def train_model(model, inputs, labels, criterion, optimizer, num_epochs, early_stopping_rounds, verbose=True):
    best_loss = float('inf')
    best_model = None
    patience = 0

    for epoch in range(num_epochs):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1

        if patience >= early_stopping_rounds:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model)
    return model


def accuracy(outputs, labels):
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


input_dim = df.shape[1]
hidden_dim = 16
output_dim = 2

model = Net(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
early_stopping_rounds = 10

train_inputs = train_inputs.float()
train_labels = train_labels.long()

model = train_model(model, train_inputs, train_labels, criterion, optimizer, num_epochs, early_stopping_rounds)

test_inputs = test_inputs.float()
test_labels = test_labels.long()
test_outputs = forward_pass(model, test_inputs)
test_loss = criterion(test_outputs, test_labels)
test_acc = accuracy(test_outputs, test_labels)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
