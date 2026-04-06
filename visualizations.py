"""Visualization helpers using Plotly and Matplotlib."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


COLORS = {
    'bg': '#0a0a1a', 'card': '#12122a', 'accent': '#6c63ff',
    'accent2': '#00d2ff', 'accent3': '#ff6b6b', 'text': '#e0e0ff',
    'grid': '#1a1a3a', 'positive': '#00e676', 'negative': '#ff5252',
    'grad1': '#6c63ff', 'grad2': '#00d2ff', 'grad3': '#ff6b6b',
    'grad4': '#ffd93d', 'grad5': '#6bcb77'
}

LAYOUT = dict(
    paper_bgcolor=COLORS['bg'], plot_bgcolor=COLORS['bg'],
    font=dict(color=COLORS['text'], family='Inter'),
    margin=dict(l=40, r=40, t=50, b=40),
)


def plot_network_arch(layer_sizes, activations=None, highlight_layer=None):
    fig = go.Figure()
    max_neurons = max(layer_sizes)
    x_spacing = 2.0
    positions = []

    for li, size in enumerate(layer_sizes):
        x = li * x_spacing
        y_start = -(size - 1) / 2.0
        layer_pos = []
        for ni in range(size):
            y = y_start + ni
            layer_pos.append((x, y))
        positions.append(layer_pos)

    # Draw connections
    for li in range(len(layer_sizes) - 1):
        for n1 in positions[li]:
            for n2 in positions[li + 1]:
                fig.add_trace(go.Scatter(
                    x=[n1[0], n2[0]], y=[n1[1], n2[1]],
                    mode='lines', line=dict(color='rgba(108,99,255,0.15)', width=1),
                    hoverinfo='skip', showlegend=False
                ))

    # Draw neurons
    for li, layer_pos in enumerate(positions):
        vals = None
        if activations and li < len(activations):
            a = activations[li]
            if a.ndim > 1:
                vals = np.mean(a, axis=0)
            else:
                vals = a

        for ni, (x, y) in enumerate(layer_pos):
            color = COLORS['accent']
            size = 28
            text = ''
            if highlight_layer is not None and li == highlight_layer:
                color = COLORS['accent2']
                size = 34
            if vals is not None and ni < len(vals):
                text = f'{vals[ni]:.2f}'

            label = ''
            if li == 0:
                label = f'x{ni+1}'
            elif li == len(layer_sizes) - 1:
                label = 'ŷ' if layer_sizes[-1] == 1 else f'ŷ{ni+1}'
            else:
                label = f'h{li},{ni+1}'

            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=size, color=color, line=dict(color='white', width=1.5),
                            opacity=0.9),
                text=label, textposition='middle center',
                textfont=dict(size=9, color='white'),
                hovertext=f'Layer {li} Neuron {ni}: {text}' if text else f'Layer {li} Neuron {ni}',
                hoverinfo='text', showlegend=False
            ))

    labels = ['Input'] + [f'Hidden {i}' for i in range(1, len(layer_sizes)-1)] + ['Output']
    for li, lbl in enumerate(labels):
        max_y = max(p[1] for p in positions[li])
        fig.add_annotation(x=li*x_spacing, y=max_y+0.8, text=lbl,
                           showarrow=False, font=dict(size=12, color=COLORS['accent2']))

    fig.update_layout(**LAYOUT, title='Network Architecture',
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'),
                      height=400)
    return fig


def plot_loss_curve(losses):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(losses)+1)), y=losses,
        mode='lines', line=dict(color=COLORS['accent'], width=2.5),
        fill='tozeroy', fillcolor='rgba(108,99,255,0.1)',
        name='Loss'
    ))
    fig.update_layout(**LAYOUT, title='Loss Over Training',
                      xaxis_title='Epoch', yaxis_title='Loss', height=350)
    return fig


def plot_decision_boundary(nn, X, y, resolution=100):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = nn.forward(grid).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        z=preds, colorscale=[[0, '#6c63ff'], [0.5, '#1a1a3a'], [1, '#00d2ff']],
        opacity=0.6, showscale=False, contours=dict(showlines=False)
    ))

    colors = [COLORS['accent3'] if yi == 0 else COLORS['accent2'] for yi in y.ravel()]
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(size=8, color=colors, line=dict(color='white', width=1)),
        showlegend=False
    ))
    fig.update_layout(**LAYOUT, title='Decision Boundary', height=400,
                      xaxis_title='x₁', yaxis_title='x₂')
    return fig


def plot_gradient_heatmap(gradients, layer_idx=0):
    g = gradients[layer_idx]
    fig = go.Figure(data=go.Heatmap(
        z=g, colorscale=[[0, '#6c63ff'], [0.5, '#0a0a1a'], [1, '#ff6b6b']],
        zmid=0, colorbar=dict(title='Gradient')
    ))
    fig.update_layout(**LAYOUT, title=f'Gradient Heatmap — Layer {layer_idx+1}',
                      xaxis_title='To Neuron', yaxis_title='From Neuron', height=350)
    return fig


def plot_gradient_flow(gradient_norms):
    norms = np.array(gradient_norms)
    fig = go.Figure()
    colors = [COLORS['grad1'], COLORS['grad2'], COLORS['grad3'], COLORS['grad4'], COLORS['grad5']]
    for li in range(norms.shape[1]):
        fig.add_trace(go.Scatter(
            x=list(range(1, norms.shape[0]+1)), y=norms[:, li],
            mode='lines', name=f'Layer {li+1}',
            line=dict(color=colors[li % len(colors)], width=2)
        ))
    fig.update_layout(**LAYOUT, title='Gradient Flow Across Layers',
                      xaxis_title='Epoch', yaxis_title='‖∇W‖', height=350,
                      yaxis_type='log')
    return fig


def plot_weight_distributions(weight_history, epoch):
    weights = weight_history[epoch]
    fig = make_subplots(rows=1, cols=len(weights),
                        subplot_titles=[f'Layer {i+1}' for i in range(len(weights))])
    colors = [COLORS['grad1'], COLORS['grad2'], COLORS['grad3'], COLORS['grad4']]
    for i, w in enumerate(weights):
        fig.add_trace(go.Histogram(
            x=w.ravel(), nbinsx=30, name=f'L{i+1}',
            marker_color=colors[i % len(colors)], opacity=0.8
        ), row=1, col=i+1)
    fig.update_layout(**LAYOUT, title=f'Weight Distributions — Epoch {epoch+1}',
                      height=300, showlegend=False)
    return fig


def plot_chain_rule_diagram():
    fig = go.Figure()
    nodes = [('L', 0, 0), ('ŷ', 2, 0), ('z', 4, 0), ('w', 6, 0.5), ('a', 6, -0.5)]
    for label, x, y in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=50, color=COLORS['accent'], opacity=0.9,
                        line=dict(color='white', width=2)),
            text=label, textposition='middle center',
            textfont=dict(size=16, color='white'), showlegend=False
        ))

    arrows = [((0, 0), (2, 0), '∂L/∂ŷ'), ((2, 0), (4, 0), '∂ŷ/∂z'),
              ((4, 0), (6, 0.5), '∂z/∂w'), ((4, 0), (6, -0.5), '∂z/∂a')]
    for (x0, y0), (x1, y1), label in arrows:
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y',
                           axref='x', ayref='y', showarrow=True,
                           arrowhead=3, arrowsize=1.5, arrowwidth=2,
                           arrowcolor=COLORS['accent2'])
        fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2 + 0.25, text=label,
                           showarrow=False, font=dict(size=11, color=COLORS['accent2']))

    fig.add_annotation(x=3, y=-1.2,
        text='<b>∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂z · ∂z/∂w</b>',
        showarrow=False, font=dict(size=14, color=COLORS['accent3']))

    fig.update_layout(**LAYOUT, title='Chain Rule — Computational Graph',
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      height=350)
    return fig


def plot_weight_update_arrows(weights_before, weights_after, gradients, layer_idx=0):
    wb = weights_before[layer_idx].ravel()[:20]
    wa = weights_after[layer_idx].ravel()[:20]
    g = gradients[layer_idx].ravel()[:20]

    fig = go.Figure()
    x = list(range(len(wb)))
    fig.add_trace(go.Bar(x=x, y=wb, name='Before', marker_color=COLORS['accent'], opacity=0.6))
    fig.add_trace(go.Bar(x=x, y=wa, name='After', marker_color=COLORS['accent2'], opacity=0.8))

    for i in range(len(g)):
        color = COLORS['positive'] if g[i] < 0 else COLORS['negative']
        fig.add_annotation(x=i, y=wa[i], text='↓' if g[i] > 0 else '↑',
                           showarrow=False, font=dict(size=14, color=color))

    fig.update_layout(**LAYOUT, title=f'Weight Updates — Layer {layer_idx+1}',
                      xaxis_title='Weight Index', yaxis_title='Value',
                      barmode='group', height=350)
    return fig
