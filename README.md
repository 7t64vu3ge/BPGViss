# 🧠 Backpropagation Visualizer (BPGViss)

An interactive, high-fidelity Streamlit application designed to visually explain the inner workings of neural network learning. This tool deconstructs the black box of backpropagation into seven digestible, interactive components.

---

## 🌟 Core Concepts Explained

The app is structured around 7 critical elements of neural network training:

1.  **🔀 Forward Pass**: Visualize how inputs flow through weight matrices and non-linear activations to produce predictions.
2.  **📉 Loss Function**: Understand Binary Cross-Entropy and how it quantifies the "distance" between prediction and reality.
3.  **⛓️ Chain Rule**: A dedicated tab for the mathematical engine—see the computational graph and how partial derivatives are calculated.
4.  **📐 Gradient Computation**: Explore gradient heatmaps for every layer to see which weights have the most impact on the error.
5.  **🔄 Backward Pass**: Watch the error signal propagate from the output layer back to the input, distributing "blame" for the loss.
6.  **🔧 Weight Updates**: See the actual parameter shifts using Gradient Descent: $w = w - \eta \cdot \nabla w$.
7.  **🌊 Learning Dynamics**: Diagnose your network health by monitoring for vanishing or exploding gradients across layers.

---

## 🚀 Tech Stack

- **Frontend**: Streamlit (Premium Dark-Mode Custom UI)
- **Charts**: Plotly (Interactive Heatmaps, Scatter Plots, and Bar Charts)
- **Engine**: Custom NumPy-based Neural Network (No deep learning frameworks used—pure math!)
- **Data**: Scikit-Learn for synthetic dataset generation

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/7t64vu3ge/BPGViss.git
   cd BPGViss
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## 📖 Walkthrough: How to Use

### 1. Configure the Network
Use the **Sidebar** to:
- Choose a **Dataset** (Moons, Circles, XOR, Spiral).
- Adjust **Architecture** (Number of hidden layers and neurons).
- Select **Activation** functions (Sigmoid, ReLU, Tanh).
- Set **Training Parameters** (Learning rate and number of epochs).

### 2. Run Training
Click the **🚀 Train Network** button. The metrics at the top will update to show the final loss and accuracy achieved.

### 3. Explore the Tabs
- **Forward Pass**: Check the decision boundary to see how the network "sees" the data.
- **Grident Heatmaps**: Look for dark or bright spots to see which connections are learning the fastest.
- **Learning Dynamics**: Use the log-scale gradient flow chart to ensure your deeper layers aren't "dying" (vanishing gradients).

### 4. Mathematical Deep Dive
Visit the **Chain Rule** tab to see the exact formula being applied at every step of the backward pass.

---

## 💡 Core Idea

> **Backpropagation** = Forward pass (predict) → Loss (measure error) → Backward pass (compute gradients) → Update weights (learn)

---
*Created with ❤️ by Antigravity*
