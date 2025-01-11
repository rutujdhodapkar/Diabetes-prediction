import streamlit as st
import matplotlib.pyplot as plt

# Define necessary variables to avoid NameError
accuracy = 0.85  # Example accuracy value
conf_matrix = [[50, 10], [5, 35]]  # Example confusion matrix
class_report = "Precision, Recall, F1-Score"  # Example classification report
train_losses = [0.6, 0.5, 0.4]  # Example train losses
test_losses = [0.7, 0.6, 0.5]  # Example test losses
train_accuracies = [0.7, 0.75, 0.8]  # Example train accuracies
test_accuracies = [0.65, 0.7, 0.75]  # Example test accuracies
total_params = 10000  # Example total parameters

# Streamlit app title
st.title('Diabetes Prediction Model')

# Display final accuracy, confusion matrix, and classification report
st.subheader('Model Evaluation')
st.write(f'**Accuracy:** {accuracy:.2f}')
st.write('**Confusion Matrix:**')
st.write(conf_matrix)
st.write('**Classification Report:**')
st.write(class_report)

# Plotting the losses and accuracies to check for overfitting
st.subheader('Training and Testing Losses')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot Losses
ax[0].plot(range(len(train_losses)), train_losses, label='Train Loss')
ax[0].plot(range(len(test_losses)), test_losses, label='Test Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Train vs Test Loss')
ax[0].legend()

# Plot Accuracies
ax[1].plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy')
ax[1].plot(range(len(test_accuracies)), test_accuracies, label='Test Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Train vs Test Accuracy')
ax[1].legend()

st.pyplot(fig)

# Print total parameters
st.write(f'**Total parameters:** {total_params}')
