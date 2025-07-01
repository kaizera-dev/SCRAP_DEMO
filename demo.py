#pip install streamlit torch matplotlib pandas "numpy<2"

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set seed for reproducibility
torch.manual_seed(0)

# Generate synthetic velocity data
def generate_velocity_data(n_samples=200, noise=0.5, delta_t=2.0):
    start_positions = torch.rand(n_samples, 2) * 100  # (x, y) positions in km
    true_velocities = torch.randn(n_samples, 2) * 7   # true velocity in km/s
    end_positions = start_positions + true_velocities * delta_t
    end_positions += noise * torch.randn(n_samples, 2)  # add noise

    features = torch.cat([start_positions, end_positions], dim=1)
    targets = true_velocities
    return features, targets

# Simple regression model
class VelocityNet(nn.Module):
    def __init__(self):
        super(VelocityNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Train model
def train_velocity_model(X, y, epochs=200, lr=0.01):
    model = VelocityNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

# Streamlit UI
st.title("🛰️ SCRAP ML Demo: Velocity Estimation from Noisy Satellite Imagery")

# Sidebar controls
samples = st.sidebar.slider("Number of Samples", 50, 1000, 200,
    help="How many object paths to simulate.")
noise = st.sidebar.slider("Sensor Noise (km)", 0.0, 2.0, 0.5,
    help="How much error to add to the second position measurement.")
delta_t = st.sidebar.slider("Time Between Snapshots (s)", 0.5, 5.0, 2.0,
    help="Interval between position detections.")
epochs = st.sidebar.slider("Training Epochs", 50, 500, 200,
    help="Number of passes through the dataset during training.")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01,
    help="How fast the model adjusts during training.")

# Generate data and train model
X, y = generate_velocity_data(samples, noise, delta_t)
model = train_velocity_model(X, y, epochs, lr)
model.eval()
with torch.no_grad():
    y_pred = model(X)

st.markdown("""
This tool simulates how our CubeSat might estimate the velocity of space debris
from two noisy position detections. We generate synthetic (x, y) positions, then
train a neural network to recover velocity using PyTorch.
""")

# Tabs for output
tab1, tab2 = st.tabs([ "📈 Velocity Prediction","📊 Data Preview"])

with tab1:
    fig, ax = plt.subplots()
    ax.quiver(X[:, 0], X[:, 1], y[:, 0], y[:, 1], color='blue', alpha=0.5, label='True Velocity')
    ax.quiver(X[:, 0], X[:, 1], y_pred[:, 0], y_pred[:, 1], color='red', alpha=0.5, label='Predicted Velocity')
    ax.set_title("True vs Predicted Velocities")
    ax.set_xlabel("X Position (km)")
    ax.set_ylabel("Y Position (km)")
    ax.legend()
    st.pyplot(fig)


with tab2:
    df = pd.DataFrame(torch.cat([X, y, y_pred], dim=1).numpy(),
                      columns=["Start X", "Start Y", "End X", "End Y", 
                               "True Vx", "True Vy", "Pred Vx", "Pred Vy"])
    st.write("First 20 samples from the simulation:")
    st.dataframe(df.head(20))
