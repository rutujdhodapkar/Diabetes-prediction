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

# Data preparation (replace this with actual dataset file if needed)
@st.cache_data
def load_and_prepare_data():
    # Load your dataset here
    # For demo, we'll create a sample dataset
    data = {
        "Age": np.random.randint(20, 70, 500),
        "Pregnancies": np.random.randint(0, 10, 500),
        "Glucose": np.random.uniform(50, 200, 500),
        "BloodPressure": np.random.uniform(60, 100, 500),
        "BMI": np.random.uniform(18.5, 40, 500),
        "DiabetesPedigreeFunction": np.random.uniform(0.1, 2, 500),
        "Insulin": np.random.uniform(15, 200, 500),
        "Outcome": np.random.randint(0, 2, 500),
    }
    df = pd.DataFrame(data)
    X = df[["Age", "Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Insulin"]]
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_and_prepare_data()

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

# User Input
st.sidebar.header("Input Features")
age = st.sidebar.slider("Age", 10, 100, 30)
pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 50.0, 200.0, 100.0)
blood_pressure = st.sidebar.slider("Blood Pressure", 40.0, 140.0, 80.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.1, 2.5, 0.5)
insulin = st.sidebar.slider("Insulin", 15.0, 300.0, 85.0)

input_data = np.array([[age, pregnancies, glucose, blood_pressure, bmi, dpf, insulin]])
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
