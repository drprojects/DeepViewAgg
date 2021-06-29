# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import os
import sys
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from torch_points3d.visualization.app.data import *

# Compute the initial state od the app
# i_sample = 10  # area5-office40
i_sample = 5  # area1-office1
dataset = get_dataset()
model = get_model(dataset)
mm_data = get_mm_sample(i_sample, dataset.test_dataset[0], model)
out = compute_plotly_visualizations(mm_data)

# Recover the image position traces in 3D visualization
img_3d_traces = out['3d']['img_traces']

# button_2d_image
fig_3d = out['3d']['figure']
visibilities_3d = [button.args[0]['visible'] for button in fig_3d.layout.updatemenus[0].buttons]

# layout['3d'] = fig_3d.layout
# fig_3d_menus = layout['3d'].updatemenus
# layout['3d'].updatemenus = None
# fig_3d.layout.updatemenus = None

# button_3d_mode = dcc.Dropdown(
#     id='point-dropdown',
#     searchable=False,
#     clearable=False,
#     options=[
#         {'label': f, 'value': str(i)} for i, f in enumerate([button.label for button in fig_3d_menus[0].buttons])],
#     value=0)

# Separate the 2D data from the figure, for faster visualization update
# at callback time
image_backgrounds = [
    out['2d_rgb']['images'],
    out['2d_pred']['images'],
    out['2d_pred_l2']['images'],
    out['2d_feat']['images'],
    out['2d_feat_l2']['images']
]

# Layout for 2D figure
layout_2d = out['2d_rgb']['figure'].layout
fig_2d_menus = layout_2d.updatemenus
layout_2d.updatemenus = None

# Initialize 2D visualization
fig_2d = go.Figure(layout=layout_2d)
fig_2d.add_trace(
    go.Image(
        z=compute_2d_back_front_visualization(image_backgrounds[0], 0, 0, 0, alpha=2).permute(1, 2, 0),
        visible=True,
        opacity=1.0,
        hoverinfo='none', ))  # disable hover info on images

button_2d_image = dcc.Dropdown(
    id='image-dropdown',
    searchable=False,
    clearable=False,
    options=[
        {'label': f, 'value': str(i)} for i, f in enumerate([button.label for button in fig_2d_menus[0].buttons])],
    value=0)

button_2d_back_1 = dcc.Dropdown(
    id='back-dropdown-1',
    searchable=False,
    clearable=False,
    options=[
        {'label': f, 'value': str(i)} for i, f in enumerate(['RGB', 'Prediction', 'Prediction L2', 'Features', 'Features L2'])],
    value=0,
)

button_2d_back_2 = dcc.Dropdown(
    id='back-dropdown-2',
    searchable=False,
    clearable=False,
    options=[
        {'label': f, 'value': str(i)} for i, f in enumerate(['RGB', 'Prediction', 'Prediction L2', 'Features', 'Features L2'])],
    value=0,
)

has_multi_front = len(fig_2d_menus) > 1
if has_multi_front:
    button_2d_front = dcc.Dropdown(
        id='front-dropdown',
        searchable=False,
        clearable=False,
        options=[
            {'label': f, 'value': str(i)} for i, f in enumerate([button.label for button in fig_2d_menus[1].buttons])],
        value=0,
    )
else:
    button_2d_front = html.Div()

button_2d_error = dcc.Dropdown(
    id='error-dropdown',
    searchable=False,
    clearable=False,
    options=[
        {'label': f, 'value': str(i)} for i, f in enumerate(['No error', 'Pointwise error', 'View-wise error'])],
    value=0,
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
style_center = {'display': 'flex', 'justify-content': 'center'}

app.layout = html.Div(children=[
    html.H1(children='Multimodal Sample', style={'textAlign': 'center', 'font-size': '30px'}),

    html.Div([
        dcc.Graph(id='graph_3d', figure=fig_3d, config={'displayModeBar': False}), ],
        style=style_center),

    html.Div([
        html.Div([button_2d_image], style={'width': '2cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([button_2d_back_1], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([button_2d_back_2], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([dcc.Slider(id='back-slider', min=0, max=1, step=0.1, value=0,)], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([button_2d_front], style={'width': '5cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([button_2d_error], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([dcc.Slider(id='alpha-slider', min=1, max=6, step=0.25, value=2,)], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),],
        style=style_center),

    html.Div([
        dcc.Graph(id='graph_2d', figure=fig_2d, config={'displayModeBar': False})],
        style=style_center),

])


@app.callback(
    Output('graph_2d', 'figure'),
    [Input('image-dropdown', 'value'),
    Input('back-dropdown-1', 'value'),
    Input('back-dropdown-2', 'value'),
    Input('back-slider', 'value'),
    Input('front-dropdown', 'value'),
    Input('error-dropdown', 'value'),
    Input('alpha-slider', 'value')])
def update_graph_2d(i_img, i_back_1, i_back_2, t_back, i_front, i_error, alpha):
    # Select background between RGB and Prediction
    images = image_backgrounds[int(i_back_1)].clone()
    for im_1, im_2 in zip(images, image_backgrounds[int(i_back_2)]):
        im_1.background = (im_1.background * (1 - float(t_back)) + im_2.background * float(t_back)).byte()

    # Select foreground and compute visualization
    fig_2d = go.Figure(layout=layout_2d)
    fig_2d.add_trace(
        go.Image(
            z=compute_2d_back_front_visualization(images, int(i_img), int(i_front), int(i_error), alpha=float(alpha)).permute(1, 2, 0),
            visible=True,
            opacity=1.0,
            hoverinfo='none', ))  # disable hover info on images

    return fig_2d





if __name__ == '__main__':
    # app.run_server(debug=True, port=8050, dev_tools_hot_reload=True)
    # app.run_server(port=8050)
    app.run_server(port=8051)
