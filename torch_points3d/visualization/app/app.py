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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from torch_points3d.visualization.app.data import OUT_3D, OUT_RGB_2D, OUT_PRED_2D, get_2d_visualization

# Recover the image position traces in 3D visualization
img_3d_traces = OUT_3D['img_traces']

# button_2d_image
fig_3d = OUT_3D['figure']
visibilities_3d = [button.args[0]['visible'] for button in fig_3d.layout.updatemenus[0].buttons]

# layout_3d = fig_3d.layout
# fig_3d_menus = layout_3d.updatemenus
# layout_3d.updatemenus = None
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
images_rgb = OUT_RGB_2D['images']
images_pred = OUT_PRED_2D['images']

# Layout for 2D figure
layout_2d = OUT_RGB_2D['figure'].layout
fig_2d_menus = layout_2d.updatemenus
layout_2d.updatemenus = None

# Initialize 2D visualization
fig_2d = go.Figure(layout=layout_2d)
fig_2d.add_trace(
    go.Image(
        z=get_2d_visualization(images_rgb, 0, 0, alpha=2).permute(1, 2, 0),
        visible=True,
        opacity=1.0,
        hoverinfo='none', ))  # disable hover info on images

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

button_2d_image = dcc.Dropdown(
    id='image-dropdown',
    searchable=False,
    clearable=False,
    options=[
        {'label': f, 'value': str(i)} for i, f in enumerate([button.label for button in fig_2d_menus[0].buttons])],
    value=0)

button_2d_back = dcc.Dropdown(
    id='back-dropdown',
    searchable=False,
    clearable=False,
    options=[
        {'label': f, 'value': str(i)} for i, f in enumerate(['RGB', 'Prediction'])],
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

n_2d_images = len(fig_2d_menus[0].buttons)
n_2d_front = len(fig_2d_menus[1].buttons) if has_multi_front else 1

style_center = {'display': 'flex', 'justify-content': 'center'}

app.layout = html.Div(children=[
    html.H1(children='Multimodal Sample', style={'textAlign': 'center'}),

    # html.Div([
    #     html.Div([button_2d_image], style={'width': '2cm', 'margin': '5px', 'display': 'inline-block'}),
    #     html.Div([button_2d_front], style={'width': '5cm', 'margin': '5px', 'display': 'inline-block'}),
    #     html.Div([dcc.Slider(id='alpha-slider', min=1, max=6, step=0.25, value=2,)], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),],
    #     style=style_center),

    html.Div([
        dcc.Graph(id='graph_3d', figure=fig_3d, config={'displayModeBar': False}),],
        style=style_center),

    html.Div([
        html.Div([button_2d_image], style={'width': '2cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([button_2d_back], style={'width': '5cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([button_2d_front], style={'width': '5cm', 'margin': '5px', 'display': 'inline-block'}),
        html.Div([dcc.Slider(id='alpha-slider', min=1, max=6, step=0.25, value=2,)], style={'width': '4cm', 'margin': '5px', 'display': 'inline-block'}),],
        style=style_center),

    html.Div([
        dcc.Graph(id='graph_2d', figure=fig_2d, config={'displayModeBar': False})],
        style=style_center),

])


@app.callback(
    Output('graph_3d', 'figure'),
    Output('graph_2d', 'figure'),
    [Input('image-dropdown', 'value'),
    Input('back-dropdown', 'value'),
    Input('front-dropdown', 'value'),
    Input('alpha-slider', 'value')])
def update_graph_2d(i_img, i_back, i_front, alpha):
    # for d, v in zip(fig_3d.data, visibilities_3d[max(int(i_front) - 1, 0)]):
    #     d.visible = v

    images = images_rgb if int(i_back) == 0 else images_pred

    fig_2d = go.Figure(layout=layout_2d)
    fig_2d.add_trace(
        go.Image(
            z=get_2d_visualization(images, int(i_img), int(i_front), alpha=float(alpha)).permute(1, 2, 0),
            visible=True,
            opacity=1.0,
            hoverinfo='none', ))  # disable hover info on images

    return fig_3d, fig_2d





if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True)
