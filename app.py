"""Backpropagation Visualizer — Streamlit App"""
import streamlit as st
import numpy as np
from nn_engine import NeuralNetwork
from datasets import get_dataset
from visualizations import (
    plot_network_arch, plot_loss_curve, plot_decision_boundary,
    plot_gradient_heatmap, plot_gradient_flow, plot_weight_distributions,
    plot_chain_rule_diagram, plot_weight_update_arrows, COLORS
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Backpropagation Visualizer", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root { --bg: #0a0a1a; --card: #12122a; --accent: #6c63ff; --accent2: #00d2ff;
        --accent3: #ff6b6b; --text: #e0e0ff; }
.stApp { background: var(--bg); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d25 0%, #12122a 100%); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--card); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: var(--text);
    font-weight: 500; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: white !important; }
.concept-card { background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(0,210,255,0.05));
    border: 1px solid rgba(108,99,255,0.3); border-radius: 16px; padding: 24px;
    margin: 12px 0; backdrop-filter: blur(10px); }
.concept-card h4 { color: var(--accent2); margin: 0 0 8px 0; }
.concept-card p { color: var(--text); opacity: 0.85; margin: 0; line-height: 1.6; }
.key-takeaway { background: linear-gradient(135deg, rgba(0,210,255,0.12), rgba(108,99,255,0.08));
    border-left: 4px solid var(--accent2); border-radius: 0 12px 12px 0;
    padding: 16px 20px; margin: 16px 0; }
.key-takeaway strong { color: var(--accent2); }
.formula-box { background: rgba(108,99,255,0.08); border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px; padding: 20px; text-align: center; margin: 12px 0; }
.hero-header { text-align: center; padding: 20px 0 10px 0; }
.hero-header h1 { background: linear-gradient(135deg, #6c63ff, #00d2ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.2rem; font-weight: 700; margin-bottom: 4px; }
.hero-header p { color: rgba(224,224,255,0.6); font-size: 1rem; }
.stat-pill { display: inline-block; background: var(--card); border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px; padding: 6px 16px; margin: 4px; font-size: 0.85rem; color: var(--accent2); }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🧠 Backpropagation Visualizer</h1>
    <p>Interactive exploration of how neural networks learn through gradient-based optimization</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    dataset_name = st.selectbox("📊 Dataset", ["Moons", "Circles", "XOR", "Spiral", "Linear"])
    n_samples = st.slider("Sample Size", 50, 500, 200, 50)

    st.markdown("---")
    st.markdown("#### 🏗️ Architecture")
    n_hidden = st.slider("Hidden Layers", 1, 4, 2)
    neurons = st.slider("Neurons per Layer", 2, 16, 4)
    activation = st.selectbox("Activation", ["sigmoid", "relu", "tanh"])

    st.markdown("---")
    st.markdown("#### 📈 Training")
    lr = st.select_slider("Learning Rate (η)", options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0], value=0.1)
    epochs = st.slider("Epochs", 10, 500, 100, 10)

    st.markdown("---")
    st.markdown("#### 🔧 Display")
    show_gradients = st.toggle("Show Gradient Overlays", value=True)
    show_decision = st.toggle("Show Decision Boundary", value=True)

    train_btn = st.button("🚀 Train Network", type="primary", use_container_width=True)

# ── Build & Train ────────────────────────────────────────────────────────────
layer_sizes = [2] + [neurons] * n_hidden + [1]
X, y = get_dataset(dataset_name, n_samples=n_samples)

if train_btn or 'nn' not in st.session_state:
    nn = NeuralNetwork(layer_sizes, lr=lr, activation=activation)
    nn.train(X, y, epochs=epochs)
    st.session_state.nn = nn
    st.session_state.X = X
    st.session_state.y = y

nn = st.session_state.nn
X = st.session_state.X
y = st.session_state.y

# ── Stats Bar ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
final_loss = nn.history['loss'][-1] if nn.history['loss'] else 0
preds = nn.forward(X)
accuracy = np.mean((preds > 0.5).astype(float) == y) * 100
with c1:
    st.metric("Final Loss", f"{final_loss:.4f}")
with c2:
    st.metric("Accuracy", f"{accuracy:.1f}%")
with c3:
    st.metric("Parameters", sum(w.size + b.size for w, b in zip(nn.weights, nn.biases)))
with c4:
    st.metric("Layers", len(layer_sizes))

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🔀 Forward Pass", "📉 Loss", "⛓️ Chain Rule",
                "📐 Gradients", "🔄 Backward Pass", "🔧 Weight Updates",
                "🌊 Learning Dynamics"])

# ── TAB 1: Forward Pass ─────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("""
    <div class="concept-card">
        <h4>Forward Pass</h4>
        <p>Data flows from the input layer through each hidden layer to produce a prediction.
        At each neuron, we compute <b>z = Wx + b</b> and then apply an activation function
        <b>a = σ(z)</b>. All intermediate activations are stored — they'll be crucial
        during backpropagation.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        last_acts = nn.history['activations'][-1] if nn.history['activations'] else None
        fig = plot_network_arch(layer_sizes, activations=last_acts)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if show_decision:
            fig_db = plot_decision_boundary(nn, X, y)
            st.plotly_chart(fig_db, use_container_width=True)

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> The forward pass transforms raw inputs into predictions
        through a series of linear transformations followed by non-linear activations. Storing
        intermediate values is essential for efficient gradient computation.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Layer Activations Detail"):
        if nn.history['activations']:
            epoch_sel = st.slider("Select Epoch (Fwd)", 1, len(nn.history['activations']),
                                  len(nn.history['activations']), key='fwd_epoch')
            acts = nn.history['activations'][epoch_sel - 1]
            for i, a in enumerate(acts):
                lbl = "Input" if i == 0 else f"Layer {i}" if i < len(acts)-1 else "Output"
                st.markdown(f"**{lbl}** — shape: {a.shape}, mean: {np.mean(a):.4f}, std: {np.std(a):.4f}")

# ── TAB 2: Loss ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("""
    <div class="concept-card">
        <h4>Loss Function</h4>
        <p>The loss function quantifies how far our predictions are from the true labels.
        We use <b>Binary Cross-Entropy</b>: L = −[y·log(ŷ) + (1−y)·log(1−ŷ)].
        A lower loss means better predictions. The gradient of this loss tells us
        the direction to adjust weights.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="formula-box">
        <span style="color:#00d2ff; font-size:1.1rem;">
        L = −1/m · Σ[ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]
        </span>
    </div>
    """, unsafe_allow_html=True)

    fig_loss = plot_loss_curve(nn.history['loss'])
    st.plotly_chart(fig_loss, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Starting Loss", f"{nn.history['loss'][0]:.4f}")
    with col2:
        reduction = ((nn.history['loss'][0] - nn.history['loss'][-1]) / nn.history['loss'][0]) * 100
        st.metric("Loss Reduction", f"{reduction:.1f}%")

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> The loss function is the objective we minimize.
        It provides the error signal that drives learning — without it, the network has
        no way to know if its predictions are good or bad.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 3: Chain Rule ───────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("""
    <div class="concept-card">
        <h4>The Chain Rule</h4>
        <p>Backpropagation is just the chain rule applied recursively. To find how a weight
        deep in the network affects the final loss, we multiply partial derivatives along the
        computational graph: <b>∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w</b>. This makes computing
        gradients for millions of parameters tractable.</p>
    </div>
    """, unsafe_allow_html=True)

    fig_chain = plot_chain_rule_diagram()
    st.plotly_chart(fig_chain, use_container_width=True)

    st.markdown("""
    <div class="formula-box">
        <span style="color:#ff6b6b; font-size:1.05rem;">
        ∂L/∂wᵢⱼ = ∂L/∂aₗ · ∂aₗ/∂zₗ · ∂zₗ/∂wᵢⱼ = δₗ · aₗ₋₁
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔗 Step-by-Step Chain Rule Walkthrough"):
        st.markdown("""
        1. **Start at the loss**: Compute ∂L/∂ŷ — how does the loss change w.r.t. prediction?
        2. **Through activation**: Multiply by ∂ŷ/∂z = σ'(z) — the activation derivative
        3. **To the weights**: Multiply by ∂z/∂w = aₗ₋₁ — the input to this layer
        4. **Recurse**: For deeper layers, keep chaining: δₗ₋₁ = (Wₗᵀ · δₗ) ⊙ σ'(zₗ₋₁)
        """)

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> The chain rule lets us decompose a complex derivative
        into a product of simpler local derivatives. This is what makes training deep networks
        computationally feasible.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 4: Gradients ────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("""
    <div class="concept-card">
        <h4>Gradient Computation</h4>
        <p>Gradients tell us how much each weight contributes to the total error.
        A large gradient means the weight has a big impact on the loss — it needs a bigger update.
        A near-zero gradient means the weight barely affects the output.</p>
    </div>
    """, unsafe_allow_html=True)

    if nn.history['gradients'] and show_gradients:
        epoch_g = st.slider("Select Epoch", 1, len(nn.history['gradients']),
                            len(nn.history['gradients']), key='grad_epoch')
        grads = nn.history['gradients'][epoch_g - 1]

        grad_cols = st.columns(min(len(grads), 3))
        for i, col in enumerate(grad_cols):
            if i < len(grads):
                with col:
                    fig_gh = plot_gradient_heatmap(grads, layer_idx=i)
                    st.plotly_chart(fig_gh, use_container_width=True)

        with st.expander("📊 Gradient Statistics"):
            for i, g in enumerate(grads):
                st.markdown(f"**Layer {i+1}** — mean: {np.mean(g):.6f}, "
                            f"std: {np.std(g):.6f}, max: {np.max(np.abs(g)):.6f}")

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> Gradients are the compass of learning. They point
        in the direction of steepest ascent — so we move in the <em>opposite</em> direction
        to minimize the loss. The magnitude tells us how confident to be in each step.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 5: Backward Pass ────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("""
    <div class="concept-card">
        <h4>Backward Pass</h4>
        <p>Starting from the output layer's error, we propagate gradients backward through
        the network. At each layer, we compute the local gradient (δ) and pass it to the
        previous layer. This is the reverse of the forward pass — error flows from
        output → input.</p>
    </div>
    """, unsafe_allow_html=True)

    if nn.history['activations']:
        last_acts = nn.history['activations'][-1]
        bw_cols = st.columns(2)
        with bw_cols[0]:
            for hl in range(len(layer_sizes)):
                fig_bw = plot_network_arch(layer_sizes, activations=last_acts,
                                           highlight_layer=len(layer_sizes)-1-hl)
                if hl == 0:
                    st.plotly_chart(fig_bw, use_container_width=True)

        with bw_cols[1]:
            st.markdown("#### Error Signal Propagation")
            if nn.history['gradients']:
                grads = nn.history['gradients'][-1]
                for i in range(len(grads)-1, -1, -1):
                    norm = np.linalg.norm(grads[i])
                    bar_width = min(norm * 200, 100)
                    st.markdown(f"""
                    <div style="margin:8px 0;">
                        <span style="color:{COLORS['accent2']}">Layer {i+1}</span>
                        <span style="color:{COLORS['text']}; opacity:0.6"> ‖∇W‖ = {norm:.6f}</span>
                        <div style="background:rgba(108,99,255,0.15); border-radius:6px; height:12px; margin-top:4px;">
                            <div style="background:linear-gradient(90deg,{COLORS['accent']},{COLORS['accent2']});
                                width:{bar_width}%; height:100%; border-radius:6px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    sel_layer = st.slider("Inspect Layer", 0, len(layer_sizes)-2, 0, key='bw_layer')
    if nn.history['activations']:
        fig_bw_arch = plot_network_arch(layer_sizes, activations=nn.history['activations'][-1],
                                         highlight_layer=sel_layer)
        st.plotly_chart(fig_bw_arch, use_container_width=True)

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> The backward pass is the heart of learning.
        It efficiently distributes blame for the prediction error to every weight in the
        network, enabling each parameter to know exactly how to change.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 6: Weight Updates ───────────────────────────────────────────────────
with tabs[5]:
    st.markdown("""
    <div class="concept-card">
        <h4>Weight Updates</h4>
        <p>Once we have gradients, we update each weight using gradient descent:
        <b>w = w − η · ∇w</b>. The learning rate (η) controls step size — too large and
        we overshoot; too small and learning is painfully slow.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
        <span style="color:#6c63ff; font-size:1.2rem;">w<sub>new</sub> = w<sub>old</sub> − η · ∂L/∂w</span>
        <br><span style="color:rgba(224,224,255,0.5); font-size:0.9rem;">where η = {lr}</span>
    </div>
    """, unsafe_allow_html=True)

    if len(nn.history['weights']) >= 2 and nn.history['gradients']:
        wu_epoch = st.slider("Epoch for Weight Update View", 1, len(nn.history['weights'])-1,
                             min(10, len(nn.history['weights'])-1), key='wu_epoch')
        wu_layer = st.slider("Layer", 0, len(nn.weights)-1, 0, key='wu_layer')

        fig_wu = plot_weight_update_arrows(
            nn.history['weights'][wu_epoch-1], nn.history['weights'][wu_epoch],
            nn.history['gradients'][wu_epoch-1], layer_idx=wu_layer
        )
        st.plotly_chart(fig_wu, use_container_width=True)

        fig_wd = plot_weight_distributions(nn.history['weights'], wu_epoch)
        st.plotly_chart(fig_wd, use_container_width=True)

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> Weight updates are where learning actually happens.
        Each update nudges parameters slightly in the direction that reduces the loss.
        Over many iterations, the network converges to a good solution.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 7: Learning Dynamics ────────────────────────────────────────────────
with tabs[6]:
    st.markdown("""
    <div class="concept-card">
        <h4>Learning Dynamics</h4>
        <p>In deep networks, gradients can <b>vanish</b> (shrink to zero in early layers)
        or <b>explode</b> (grow uncontrollably). This diagram shows gradient magnitudes
        across layers and epochs — revealing potential training instabilities.</p>
    </div>
    """, unsafe_allow_html=True)

    gnorms = nn.get_gradient_norms()
    if gnorms:
        fig_gf = plot_gradient_flow(gnorms)
        st.plotly_chart(fig_gf, use_container_width=True)

        gnorms_arr = np.array(gnorms)
        ld_cols = st.columns(len(nn.weights))
        for i, col in enumerate(ld_cols):
            with col:
                final_norm = gnorms_arr[-1, i]
                initial_norm = gnorms_arr[0, i]
                ratio = final_norm / (initial_norm + 1e-15)
                status = "🟢 Healthy" if 0.1 < ratio < 10 else ("🔴 Vanishing" if ratio < 0.1 else "🟠 Exploding")
                st.markdown(f"""
                <div style="background:{COLORS['card']}; border-radius:12px; padding:16px; text-align:center;
                    border:1px solid rgba(108,99,255,0.2);">
                    <div style="color:{COLORS['accent2']}; font-size:0.85rem;">Layer {i+1}</div>
                    <div style="color:white; font-size:1.3rem; font-weight:600;">{final_norm:.4f}</div>
                    <div style="font-size:0.8rem;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("📚 Understanding Gradient Problems"):
            st.markdown("""
            **Vanishing Gradients** — Common with sigmoid/tanh in deep networks. Early layers
            receive tiny gradients, effectively stopping learning. Solutions: ReLU, skip
            connections, batch normalization.

            **Exploding Gradients** — Gradients grow exponentially through layers. Leads to
            unstable training (NaN losses). Solutions: gradient clipping, careful initialization,
            lower learning rates.

            **Healthy Gradients** — Gradient norms stay within a reasonable range across all
            layers and epochs. This is the sweet spot for efficient learning.
            """)

    st.markdown("""
    <div class="key-takeaway">
        <strong>💡 Key Takeaway:</strong> Monitoring gradient flow is essential for debugging
        deep networks. Vanishing or exploding gradients are the most common causes of
        training failure, and understanding them is key to designing better architectures.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:20px; opacity:0.5;">
    <p>Backpropagation = Forward Pass (predict) → Loss (measure error) →
    Backward Pass (compute gradients) → Update Weights (learn)</p>
    <p style="font-size:0.8rem;">Built with Streamlit & Plotly</p>
</div>
""", unsafe_allow_html=True)
