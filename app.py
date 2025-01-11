import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set your device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit interface for file upload
st.title("Diabetes Prediction Model")
st.write("Upload a CSV file to train and test the model.")

uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)

    # Display dataset
    st.write(df.head())

    # Process the dataset
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

    # Define the neural network model with layers large enough to achieve 800 million parameters
    class DiabetesModel(nn.Module):
        def __init__(self):
            super(DiabetesModel, self).__init__()

            # Scaled model to achieve ~800 million parameters
            self.fc1 = nn.Linear(7, 10000)  # First layer with 10,000 neurons
            self.dropout1 = nn.Dropout(p=0.5)
            self.fc2 = nn.Linear(10000, 5000)  # Second layer with 5,000 neurons
            self.dropout2 = nn.Dropout(p=0.5)
            self.fc3 = nn.Linear(5000, 2500)  # Third layer with 2,500 neurons
            self.fc4 = nn.Linear(2500, 1000)   # Fourth layer with 1,000 neurons
            self.fc5 = nn.Linear(1000, 500)    # Fifth layer with 500 neurons
            self.fc6 = nn.Linear(500, 100)     # Sixth layer with 100 neurons
            self.fc7 = nn.Linear(100, 1)       # Output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            x = torch.relu(self.fc6(x))
            x = torch.sigmoid(self.fc7(x))  # Sigmoid for binary classification
            return x

    # Initialize the model
    model = DiabetesModel().to(device)

    # Define loss function and optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight_decay for L2 regularization
    criterion = nn.BCELoss()

    # Track training and test losses
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Training loop with Early Stopping
    epochs = 100
    best_test_loss = float('inf')
    patience = 10  # Early stopping patience (epochs without improvement)
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Forward pass on training data
        y_pred_train = model(X_train_tensor).squeeze()
        loss_train = criterion(y_pred_train, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Track train loss and accuracy
        train_losses.append(loss_train.item())
        train_accuracy = accuracy_score(y_train_tensor.cpu(), (y_pred_train > 0.5).float().cpu())
        train_accuracies.append(train_accuracy)

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor).squeeze()
            loss_test = criterion(y_pred_test, y_test_tensor)

            # Track test loss and accuracy
            test_losses.append(loss_test.item())
            test_accuracy = accuracy_score(y_test_tensor.cpu(), (y_pred_test > 0.5).float().cpu())
            test_accuracies.append(test_accuracy)

        # Check early stopping
        if loss_test < best_test_loss:
            best_test_loss = loss_test
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1

        if patience_counter >= patience:
            st.write(f'Early stopping at epoch {epoch+1}')
            break

        if epoch % 10 == 0:
            st.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}')

    # Evaluate final accuracy on test data
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).squeeze()
        y_pred_test = (y_pred_test > 0.5).float()  # Apply threshold of 0.5

        accuracy = accuracy_score(y_test_tensor.cpu(), y_pred_test.cpu())
        conf_matrix = confusion_matrix(y_test_tensor.cpu(), y_pred_test.cpu())
        class_report = classification_report(y_test_tensor.cpu(), y_pred_test.cpu())

        st.write(f'Accuracy: {accuracy:.2f}')
        st.write('Confusion Matrix:')
        st.write(conf_matrix)
        st.write('Classification Report:')
        st.write(class_report)

    # Plotting the losses and accuracies to check for overfitting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Losses
    axes[0].plot(range(len(train_losses)), train_losses, label='Train Loss')
    axes[0].plot(range(len(test_losses)), test_losses, label='Test Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train vs Test Loss')
    axes[0].legend()

    # Plot Accuracies
    axes[1].plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
    axes[1].plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train vs Test Accuracy')
    axes[1].legend()

    st.pyplot(fig)

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    st.write(f'Total parameters: {total_params}')
