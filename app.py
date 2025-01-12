import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Define the neural network model
class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
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

# Streamlit Title
st.title("Diabetes Prediction App")
st.write("Predict the likelihood of diabetes based on user input.")

# Data preparation (using your diabetes.csv dataset)
@st.cache_data
def load_and_prepare_data():
    # Load the dataset (make sure the file is in the correct directory)
    df = pd.read_csv("diabetes.csv")
    
    # Select features and target variable
    X = df[["Age", "Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Insulin"]]
    y = df["Outcome"]
    
    # Split the data into training and testing sets
    return df, train_test_split(X, y, test_size=0.2, random_state=42)

# Load data and split
df, (X_train, X_test, y_train, y_test) = load_and_prepare_data()

# Display the first 5 rows from the dataset
st.subheader("First 5 Rows from the Dataset")
st.write("Here are the first 5 rows from the dataset:")
st.write(df.head())

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Convert to PyTorch tensors
device = torch.device("cpu")
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# Train the model if not already trained
model = DiabetesModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

if "model_trained" not in st.session_state:
    st.write("Training the model. Please wait...")
    for epoch in range(10):  # Train for fewer epochs for simplicity
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor).squeeze()
        loss = criterion(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "model.pth")
    st.session_state["model_trained"] = True
    st.write("Model trained and saved!")

# Load the model for prediction
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# User Input (Text Boxes for User Input)
st.sidebar.header("Input Features")
age = st.sidebar.text_input("Age", "30")  # Default value is "30"
pregnancies = st.sidebar.text_input("Pregnancies", "1")  # Default value is "1"
glucose = st.sidebar.text_input("Glucose", "100")  # Default value is "100"
blood_pressure = st.sidebar.text_input("Blood Pressure", "80")  # Default value is "80"
bmi = st.sidebar.text_input("BMI", "25")  # Default value is "25"
dpf = st.sidebar.text_input("Diabetes Pedigree Function", "0.5")  # Default value is "0.5"
insulin = st.sidebar.text_input("Insulin", "85")  # Default value is "85"

# Convert user input to numeric values
try:
    input_data = np.array([[float(age), float(pregnancies), float(glucose), 
                            float(blood_pressure), float(bmi), float(dpf), float(insulin)]]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Prediction
    if st.button("Predict"):
        with torch.no_grad():
            prediction = model(input_tensor).item()
        if prediction > 0.5:
            st.error(f"High likelihood of diabetes! Probability: {prediction:.2f}")
        else:
            st.success(f"Low likelihood of diabetes. Probability: {prediction:.2f}")
except ValueError:
    st.error("Please enter valid numeric inputs for all fields.")
