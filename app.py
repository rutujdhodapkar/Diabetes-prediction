import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Set your device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming df is your dataset and you have already processed the data as in the original code
df = pd.read_csv('diabetes.csv')  # Load the dataset (make sure the file is available)

# Preprocess the data
class_0_df = df[df['Outcome'] == 0]
class_1_df = df[df['Outcome'] == 1]

class_0_df = class_0_df.sample(268)
data = pd.concat([class_0_df, class_1_df])

col_name = ['Age', 'Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Insulin']
X = data[col_name]
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# Simple Neural Network Model (From original code)
class SimpleDiabetesModel(nn.Module):
    def __init__(self):
        super(SimpleDiabetesModel, self).__init__()
        self.fc1 = nn.Linear(7, 3500)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(3500, 1500)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1500, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Large Neural Network Model (From second code)
class LargeDiabetesModel(nn.Module):
    def __init__(self):
        super(LargeDiabetesModel, self).__init__()
        self.fc1 = nn.Linear(7, 50000)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(50000, 25000)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(25000, 12500)
        self.fc4 = nn.Linear(12500, 5000)
        self.fc5 = nn.Linear(5000, 2000)
        self.fc6 = nn.Linear(2000, 500)
        self.fc7 = nn.Linear(500, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        return x

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Choose a model to predict the likelihood of diabetes based on user input.")
model_choice = st.selectbox("Select Model", ["Simple Model", "Large Model"])

# Model initialization based on user selection
if model_choice == "Simple Model":
    model = SimpleDiabetesModel().to(device)
    model_name = "Simple Model"
elif model_choice == "Large Model":
    model = LargeDiabetesModel().to(device)
    model_name = "Large Model"

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Weight decay for L2 regularization
criterion = nn.BCELoss()

# Training process (simplified version for demonstration)
@st.cache_data
def train_model():
    epochs = 100
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        y_pred_train = model(X_train_tensor).squeeze()
        loss_train = criterion(y_pred_train, y_train_tensor)
        
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        train_losses.append(loss_train.item())
        train_accuracy = accuracy_score(y_train_tensor.cpu(), (y_pred_train > 0.5).float().cpu())
        train_accuracies.append(train_accuracy)
        
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor).squeeze()
            loss_test = criterion(y_pred_test, y_test_tensor)
        
        test_losses.append(loss_test.item())
        test_accuracy = accuracy_score(y_test_tensor.cpu(), (y_pred_test > 0.5).float().cpu())
        test_accuracies.append(test_accuracy)
    
    return train_losses, test_losses, train_accuracies, test_accuracies

if st.button(f"Train {model_name}"):
    st.write(f"Training the {model_name}...")
    train_losses, test_losses, train_accuracies, test_accuracies = train_model()
    st.write(f"{model_name} trained successfully!")
    
    # Total Parameters
    total_params = sum(p.numel() for p in model.parameters())
    st.write(f"Total Parameters in {model_name}: {total_params}")

    # Plot Losses and Accuracies
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()

    plt.tight_layout()
    st.pyplot()

# Results Evaluation
if st.button(f"Evaluate {model_name}"):
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze()
        y_pred_test = (y_pred_test > 0.5).float()
        accuracy = accuracy_score(y_test_tensor.cpu(), y_pred_test.cpu())
        conf_matrix = confusion_matrix(y_test_tensor.cpu(), y_pred_test.cpu())
        class_report = classification_report(y_test_tensor.cpu(), y_pred_test.cpu())

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        st.write("Classification Report:")
        st.write(class_report)
