import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import streamlit as st

# Set your device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your dataset (replace with your actual dataset)
# Example dataset loading (modify as needed)
df = pd.read_csv('your_dataset.csv')  # Ensure the file is in the Streamlit cloud folder

# Preprocess data
class_0_df = df[df['Outcome'] == 0]
class_1_df = df[df['Outcome'] == 1]

# Balance the classes
class_0_df = class_0_df.sample(len(class_1_df), random_state=42)
data = pd.concat([class_0_df, class_1_df])

# Select features and target
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

# Define the neural network model
class DiabetesModelBig(nn.Module):
    def __init__(self):
        super(DiabetesModelBig, self).__init__()
        self.fc1 = nn.Linear(7, 20000)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(20000, 15000)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(15000, 10000)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(10000, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Initialize the model
model = DiabetesModelBig().to(device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
criterion = nn.BCELoss()

# Track metrics
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Training loop
epochs = 10  # Reduce for Streamlit runtime
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    y_pred_train = model(X_train_tensor).squeeze()
    loss_train = criterion(y_pred_train, y_train_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # Evaluate training performance
    train_losses.append(loss_train.item())
    train_accuracy = accuracy_score(y_train_tensor.cpu(), (y_pred_train > 0.5).float().cpu())
    train_accuracies.append(train_accuracy)

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze()
        loss_test = criterion(y_pred_test, y_test_tensor)
        
        test_losses.append(loss_test.item())
        test_accuracy = accuracy_score(y_test_tensor.cpu(), (y_pred_test > 0.5).float().cpu())
        test_accuracies.append(test_accuracy)

    if epoch % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")

# Evaluate final performance
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_tensor).squeeze()
    y_pred_test = (y_pred_test > 0.5).float()
    
    accuracy = accuracy_score(y_test_tensor.cpu(), y_pred_test.cpu())
    conf_matrix = confusion_matrix(y_test_tensor.cpu(), y_pred_test.cpu())
    class_report = classification_report(y_test_tensor.cpu(), y_pred_test.cpu())

# Streamlit Deployment
st.title("Diabetes Prediction with 600M Parameters!")
st.write("Accuracy on Test Data:", accuracy)
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Classification Report:")
st.text(class_report)

# Input for predictions
st.sidebar.title("Input New Data")
inputs = []
for col in col_name:
    inputs.append(st.sidebar.number_input(f"Enter {col}:"))

# Predict new data
if st.sidebar.button("Predict"):
    new_data = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)
    prediction = model(new_data).item()
    st.sidebar.write("Prediction:", "Diabetic" if prediction > 0.5 else "Non-Diabetic")
