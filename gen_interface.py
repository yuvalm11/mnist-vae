import io
import base64
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from PIL import Image

import torch
from vae_model import VAE
from tiny_cnn_model import TinyCNN


vae = VAE(latent_dim=2)
vae.load_state_dict(torch.load('vae.pth'))
vae.eval()

tiny_cnn = torch.load('tiny_mnist_cnn.pth')

def latent_to_mnist(latent_point):
    latent_point = torch.tensor(latent_point).float()
    latent_point = latent_point.unsqueeze(0)
    img = vae.decode(latent_point)
    class_idx = torch.argmax(tiny_cnn(1-img)).item()
    img = img.squeeze().detach().numpy()
    img = (img * 255).clip(0, 255)
    img = img.astype(np.uint8)
    return img, class_idx

n_points = 2000
x, y = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_points).T
x = x.clip(-3, 3)
y = y.clip(-3, 3)

class_colors = [
    "blue", "red", "pink", "orange", "green", 
    "purple", "cyan", "goldenrod", "darkgreen", "darkred"
]

point_colors = ["lightgray"] * n_points
point_size = [10] * n_points

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(
        id='gaussian-plot',
        config={'displayModeBar': False},
        style={'height': '600px'},
    ),
    html.Div(
        id='hover-image',
        style={
            'position': 'absolute',
            'pointerEvents': 'none',
            'backgroundColor': 'white',
            'padding': '10px',
            'border': '1px solid black',
            'borderRadius': '5px',
            'display': 'none',
            'zIndex': 1000
        }
    )
],
style={
    'backgroundColor': '#2d2d2d',
    'height': 'calc(100vh - 15px)',
    }
)

figure = go.Figure()
figure.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=10, color=point_colors),
    hoverinfo='none',
    name='Gaussian Points'
))
figure.update_layout(
    showlegend=False,
    height=758,
    width=758,
    plot_bgcolor='#2d2d2d',
    paper_bgcolor='#2d2d2d',
    font=dict(color='white'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

@app.callback(
    [Output('gaussian-plot', 'figure'),
     Output('hover-image', 'style'),
     Output('hover-image', 'children')],
    [Input('gaussian-plot', 'hoverData')]
)
def update_hover(hover_data):
    global point_colors

    if hover_data:
        point_index = hover_data['points'][0]['pointIndex']
        latent_point = [x[point_index], y[point_index]]

        img, class_idx = latent_to_mnist(latent_point)

        point_colors[point_index] = class_colors[class_idx]
        figure.update_traces(marker=dict(color=point_colors))

        img_pil = Image.fromarray(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        style = {
            'position': 'absolute',
            'pointerEvents': 'none',
            'backgroundColor': '#2d2d2d',
            'padding': '10px',
            'top': '110px',
            'left': '730px',
            'display': 'block',
            'zIndex': 1000
        }
        children = html.Img(src=f"data:image/png;base64,{img_str}", style={'height': '490px', 'width': '490px'})
        children = html.Div([html.H3(f"Predicted Class: {class_idx}"), children], style={'textAlign': 'center', 'fontFamily': 'Courier New', 'color': 'white'})

        return figure, style, children

    return figure, {'display': 'none'}, None

if __name__ == '__main__':
    app.run_server(debug=True)
