from torch_points3d.core.multimodal.data import MMData
from torch_points3d.core.multimodal.image import SameSettingImageData, \
    ImageData
from torch_geometric.transforms import FixedPoints
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.core.data_transform.multimodal.projection import \
    pose_to_rotation_matrix_numba
from torch_points3d.core.data_transform.multimodal.image import \
    SelectMappingFromPointId
import os.path as osp
import plotly.graph_objects as go
import numpy as np
import torch
from itertools import chain


# TODO: To go further with ipwidgets :
#  - https://plotly.com/python/figurewidget-app/
#  - https://ipywidgets.readthedocs.io/en/stable/

def rgb_to_plotly_rgb(rgb):
    """Convert torch.Tensor of float RGB values in [0, 1] to
    plotly-friendly RGB format.
    """
    assert isinstance(rgb, torch.Tensor) and rgb.max() <= 1.0 and rgb.dim() <= 2

    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)

    return [f"rgb{tuple(x)}" for x in (rgb * 255).int().numpy()]


def feats_to_rgb(feats):
    """Convert features of the format M x N with N>=1 to an M x 3
    tensor with values in [0, 1 for RGB visualization].
    """
    if feats.dim() == 1:
        feats = feats.unsqueeze(1)
    elif feats.dim() > 2:
        raise NotImplementedError

    if feats.shape[1] == 3:
        color = feats
        
    elif feats.shape[1] == 1:
        # If only 1 feature is found convert to a 3-channel
        # repetition for black-and-white visualization.
        color = feats.repeat_interleave(3, 1)
        
    elif feats.shape[1] == 2:
        # If 2 features are found, add an extra channel.
        color = torch.cat([feats, torch.ones(feats.shape[0], 1)], 1)
    
    elif feats.shape[1] > 3:
        # If more than 3 features or more are found, project
        # features to a 3-dimensional space using PCA
        x_centered = feats - feats.mean(axis=0)
        cov_matrix = x_centered.T.mm(x_centered) / len(x_centered)
        _, eigenvectors = torch.symeig(cov_matrix, eigenvectors=True)
        color = x_centered.mm(eigenvectors[:, -3:])

    # Unit-normalize the features in a hypercube of shared scale
    # for nicer visualizations
    color = color - color.min(dim=0).values.view(1, -1) \
        if color.max() != color.min()\
        else color
    color = color / (color.max(dim=0).values.view(1, -1) + 1e-6)
    
    return color


def visualize_3d(mm_data, class_names=None, class_colors=None,
        class_opacities=None, figsize=800, width=None, height=None, voxel=0.1,
        max_points=100000, pointsize=5, error_color=None, **kwargs):
    """3D data interactive visualization tools."""
    assert isinstance(mm_data, MMData)

    # 3D visualization modes
    modes = {'name': [], 'key': [], 'num_traces': []}

    # Make copies of the data and images to be modified in this scope
    data = mm_data.data.clone()
    images = mm_data.modalities['image'].clone()

    # Convert images to ImageData for convenience
    if isinstance(images, SameSettingImageData):
        images = ImageData([images])

    # Subsample to limit the drawing time
    data = GridSampling3D(voxel)(data)
    if data.num_nodes > max_points:
        data = FixedPoints(max_points, replace=False, allow_duplicates=False)(
            data)

    # Subsample the mappings accordingly
    transform = SelectMappingFromPointId()
    data, images = transform(data, images)

    # Round to the cm for cleaner hover info
    data.pos = (data.pos * 100).round() / 100
    for im in images:
        im.pos = (im.pos * 100).round() / 100

    # Prepare figure
    width = width if width and height else figsize
    height = height if width and height else int(figsize / 2)
    margin = int(0.02 * min(width, height))
    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(aspectmode='data', ),  # preserve aspect ratio
        margin=dict(l=margin, r=margin, b=margin, t=margin),)
    fig = go.Figure(layout=layout)

    # Draw a trace for RGB 3D point cloud
    fig.add_trace(
        go.Scatter3d(
            name='RGB',
            x=data.pos[:, 0],
            y=data.pos[:, 1],
            z=data.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=rgb_to_plotly_rgb(data.rgb),),
            hoverinfo='x+y+z',
            showlegend=False,
            visible=True,))
    modes['name'].append('RGB')
    modes['key'].append('rgb')
    modes['num_traces'].append(1)

    # Draw a trace for labeled 3D point cloud
    if getattr(data, 'y', None) is not None:
        y = data.y.numpy()
        n_classes = int(y.max() + 1)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        if class_colors is None:
            class_colors = [None] * n_classes
        elif not isinstance(class_colors[0], str):
            class_colors = [f"rgb{tuple(x)}" for x in class_colors]
        if class_opacities is None:
            class_opacities = [1.0] * n_classes

        n_y_traces = 0
        for label in np.unique(y):
            indices = np.where(y == label)[0]

            fig.add_trace(
                go.Scatter3d(
                    name=class_names[label],
                    opacity=class_opacities[label],
                    x=data.pos[indices, 0],
                    y=data.pos[indices, 1],
                    z=data.pos[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=pointsize,
                        color=class_colors[label],),
                    visible=False,))
            n_y_traces += 1  # keep track of the number of traces

        modes['name'].append('Labels')
        modes['key'].append('y')
        modes['num_traces'].append(n_y_traces)

    # Draw a trace for predicted labels 3D point cloud
    if getattr(data, 'pred', None) is not None:
        pred = data.pred.numpy()
        n_classes = int(pred.max() + 1)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]
        if class_colors is None:
            class_colors = [None] * n_classes
        elif not isinstance(class_colors[0], str):
            class_colors = [f"rgb{tuple(x)}" for x in class_colors]
        if class_opacities is None:
            class_opacities = [1.0] * n_classes

        n_pred_traces = 0
        for label in np.unique(pred):
            indices = np.where(pred == label)[0]
            fig.add_trace(
                go.Scatter3d(
                    name=class_names[label],
                    opacity=class_opacities[label],
                    x=data.pos[indices, 0],
                    y=data.pos[indices, 1],
                    z=data.pos[indices, 2],
                    mode='markers',
                    marker=dict(
                        size=pointsize,
                        color=class_colors[label], ),
                    visible=False, ))
            n_pred_traces += 1  # keep track of the number of traces

        modes['name'].append('Predictions')
        modes['key'].append('pred')
        modes['num_traces'].append(n_pred_traces)

    # Draw a trace for 3D point cloud of number of images seen
    n_seen = sum([im.mappings.pointers[1:] - im.mappings.pointers[:-1]
                  for im in images])
    fig.add_trace(
        go.Scatter3d(
            name='Times seen',
            x=data.pos[:, 0],
            y=data.pos[:, 1],
            z=data.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=n_seen,
                colorscale='spectral',
                colorbar=dict(
                    thickness=10, len=0.66, tick0=0,
                    dtick=max(1, int(n_seen.max() / 10.)),),),
            hovertext=[f"seen: {n}" for n in n_seen],
            hoverinfo='x+y+z+text',
            showlegend=False,
            visible=False,))
    modes['name'].append('Times seen')
    modes['key'].append('n_seen')
    modes['num_traces'].append(1)

    # Draw a trace for position-colored 3D point cloud
    radius = torch.norm(data.pos - data.pos.mean(dim=0), dim=1).max()
    data.pos_rgb = (data.pos - data.pos.mean(dim=0)) / (2 * radius) + 0.5
    fig.add_trace(
        go.Scatter3d(
            name='Position RGB',
            x=data.pos[:, 0],
            y=data.pos[:, 1],
            z=data.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=rgb_to_plotly_rgb(data.pos_rgb),),
            hoverinfo='x+y+z',
            showlegend=False,
            visible=False,))
    modes['name'].append('Position RGB')
    modes['key'].append('position_rgb')
    modes['num_traces'].append(1)

    # Draw a trace for 3D point cloud features
    if getattr(data, 'x', None) is not None:
        # Recover the features and convert them to an RGB format for 
        # visualization.
        data.feat_3d = feats_to_rgb(data.x)
        fig.add_trace(
            go.Scatter3d(
                name='Features 3D',
                x=data.pos[:, 0],
                y=data.pos[:, 1],
                z=data.pos[:, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=rgb_to_plotly_rgb(data.feat_3d), ),
                hoverinfo='x+y+z',
                showlegend=False,
                visible=False, ))
        modes['name'].append('Features 3D')
        modes['key'].append('x')
        modes['num_traces'].append(1)

    # Add a trace for prediction errors
    if getattr(data, 'y', None) is not None \
            and getattr(data, 'pred', None) is not None:
        indices = np.where(data.pred.numpy() != data.y.numpy())[0]
        error_color = f"rgb{tuple(error_color)}" \
            if error_color is not None else 'rgb(0, 0, 0)'
        fig.add_trace(
            go.Scatter3d(
                name='Errors',
                opacity=1.0,
                x=data.pos[indices, 0],
                y=data.pos[indices, 1],
                z=data.pos[indices, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=error_color, ),
                showlegend=False,
                visible=False, ))
        modes['name'].append('Errors')
        modes['key'].append('error')
        modes['num_traces'].append(1)

    # Draw image positions
    if images.num_settings > 1:
        image_xyz = torch.cat([im.pos for im in images]).numpy()
        image_opk = torch.cat([im.opk for im in images]).numpy()
    else:
        image_xyz = images[0].pos.numpy()
        image_opk = images[0].opk.numpy()
    if len(image_xyz.shape) == 1:
        image_xyz = image_xyz.reshape((1, -1))
    for i, (xyz, opk) in enumerate(zip(image_xyz, image_opk)):

        # Draw image coordinate system axes
        arrow_length = 0.1
        for v, color in zip(np.eye(3), ['red', 'green', 'blue']):
            v = xyz + pose_to_rotation_matrix_numba(opk).dot(v * arrow_length)
            fig.add_trace(
                go.Scatter3d(
                    x=[xyz[0], v[0]],
                    y=[xyz[1], v[1]],
                    z=[xyz[2], v[2]],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=pointsize + 1),
                    showlegend=False,
                    hoverinfo='none',
                    visible=True,))

        # Draw image position as ball
        fig.add_trace(
            go.Scatter3d(
                name=f"Image {i}",
                x=[xyz[0]],
                y=[xyz[1]],
                z=[xyz[2]],
                mode='markers+text',
                marker=dict(
                    line_width=2,
                    size=pointsize + 4,),
                text=f"<b>{i}</b>",
                textposition="bottom center",
                textfont=dict(
                    size=16),
                hoverinfo='x+y+z+name',
                showlegend=False,
                visible=True,))

    # Traces visibility for interactive point cloud coloring
    def trace_visibility(mode):
        visibilities = np.array([d.visible for d in fig.data], dtype='bool')

        # Traces visibility for interactive point cloud coloring
        i_mode = modes['key'].index(mode)
        a = sum(modes['num_traces'][:i_mode])
        b = sum(modes['num_traces'][:i_mode+1])
        n_traces = sum(modes['num_traces'])

        visibilities[:n_traces] = False
        visibilities[a:b] = True

        return [{"visible": visibilities.tolist()}]

    # Create the buttons that will serve for toggling trace visibility
    updatemenus = [
        dict(
            buttons=[dict(label=name, method='update', args=trace_visibility(key))
               for name, key in zip(modes['name'], modes['key']) if key != 'error'],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02,),
        dict(
            buttons=[dict(method='restyle',
                label='Error',
                visible=True,
                args=[{'visible': True,}, [sum(modes['num_traces'][:modes['key'].index('error')])]],
                args2=[{'visible': False,}, [sum(modes['num_traces'][:modes['key'].index('error')])]],)],
            pad={'r': 10, 't': 10},
            showactive=False,
            type='buttons',
            xanchor='left',
            x=1.02,
            yanchor='top',
            y=1.02, ),]
    fig.update_layout(updatemenus=updatemenus)

    # Place the legend on the left
    fig.update_layout(
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99))

    return fig


def visualize_2d(mm_data, figsize=800, width=None, height=None, alpha=3,
        color_mode='light', class_colors=None, show_point_error=False,
        show_view_error=False, error_color=None, **kwargs):
    """2D data interactive visualization tools."""
    assert isinstance(mm_data, MMData)

    # Make copies of the data and images to be modified in this scope
    data = mm_data.data.clone()
    images = mm_data.modalities['image'].clone()

    # Convert images to ImageData for convenience
    if isinstance(images, SameSettingImageData):
        images = ImageData([images])
    
    # Fallback to 'light' visualization mode if provided mode is
    # unknown
    MODES_2D = ['light', 'rgb', 'pos', 'y', 'feat_3d', 'feat_proj', 'pred']
    color_mode = color_mode if color_mode in MODES_2D else 'light'

    # Sanity checks
    no_3d_x = getattr(data, 'x', None) is None
    no_3d_y = getattr(data, 'y', None) is None
    no_3d_pred = getattr(data, 'pred', None) is None
    no_2d_x = any([getattr(im, 'x', None) is None for im in images])
    no_2d_pred = any([getattr(im, 'pred', None) is None for im in images])
    no_map_feat = any([im.mappings.features is None for im in images])
    if color_mode == 'y' and class_colors is None:
        raise ValueError("Mode 'y' requires 'class_colors'.")
    elif color_mode == 'feat_3d' and no_2d_x:
        raise ValueError("Mode 'feat_3d' requires images to carry an 'x' attribute.")
    elif color_mode == 'feat_proj' and no_map_feat:
        raise ValueError("Mode 'feat_proj' requires mappings to carry a 'features' attribute.")
    elif color_mode == 'pred' and no_2d_pred:
        raise ValueError("Mode 'pred' requires images to carry a 'pred' attribute.")
    if show_point_error and (no_3d_y or no_3d_pred):
        raise ValueError("'show_point_error' requires points to carry 'y' and 'pred' attributes.")
    if show_view_error and (no_3d_y or no_2d_pred):
        raise ValueError("'show_view_error' requires points to carry 'y' attributes and images to carry 'pred' attributes.")
    if show_point_error and show_view_error:
        raise ValueError("Please choose either 'show_point_error' or 'show_point_error', but not both.")
    error_color = torch.ByteTensor(error_color).view(1, 3) \
        if error_color is not None else torch.zeros(1, 3, dtype=torch.uint8)

    # Convert 2D predictions to RGB colors, if need be
    if color_mode == 'pred':
        for im in images:
            pred = im.pred

            # If logits, convert to labels1
            if pred.is_floating_point():
                pred = pred.argmax(dim=1).long()

            # If class labels are passed in (N, 1, H, W) tensor, convert
            # to (N, H, W) long tensor
            if len(pred.shape) == 4 and pred.shape[1] == 1:
                pred = pred.squeeze(1).long()

            # Save the labels to pred attribute
            im.pred = pred

            # Convert int labels to RGB colors
            if len(pred.shape) == 3:
                pred = torch.ByteTensor(class_colors)[pred.long()].permute(0, 3, 1, 2)
            elif not pred.shape[1] == 3:
                raise ValueError(
                    "'pred' mode assumes image's 'x' attribute to either carry"
                    " class logits or class labels.")

            # Save the RGB-converted labels to x attribute
            im.x = pred

    # Load images if need be
    images = ImageData([im.load() if im.x is None else im for im in images])

    # Convert image features to [0, 1] colors, if need be. All images 
    # must be handled at once, in case we need to PCA the features in a 
    # common projective space.
    if images[0].x.is_floating_point():
        shapes = [im.x.shape for im in images]
        sizes = [s[0] * s[2] * s[3] for s in shapes]
        feats = torch.cat([
            im.x.permute(1, 0, 2, 3).reshape(s[1], -1).T
            for im, s in zip(images, shapes)], dim=0)
        colors = feats_to_rgb(feats)
        colors = [x.T.reshape(3, s[0], s[2], s[3]).permute(1, 0, 2, 3) 
                 for x, s in zip(colors.split(sizes), shapes)]
        for pred, im in zip(colors, images):
            im.x = (pred * 255).byte()

    for im in images:
        # Color the images where points are projected and darken the 
        # rest
        im.x = (im.x.float() / alpha).floor().type(torch.uint8)

        # Get the mapping of all points in the sample
        idx = im.mappings.feature_map_indexing

        if color_mode in ['light', 'pred']:
            # Set mapping mask back to original lighting
            color = torch.full((3,), alpha, dtype=torch.uint8)
            color = im.x[idx] * color

        elif color_mode == 'rgb':
            # Set mapping mask to point cloud RGB colors
            color = (data.rgb * 255).type(torch.uint8)
            color = color.repeat_interleave(
                im.mappings.pointers[1:] - im.mappings.pointers[:-1],
                dim=0)
            color = color.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1],
                dim=0)

        elif color_mode == 'pos':
            # Set mapping mask to point cloud positional RGB colors
            radius = torch.norm(
                data.pos - data.pos.mean(dim=0), dim=1).max()
            color = ((data.pos - data.pos.mean(dim=0))
                     / (2 * radius) * 255 + 127).type(torch.uint8)
            color = color.repeat_interleave(
                im.mappings.pointers[1:] - im.mappings.pointers[:-1],
                dim=0)
            color = color.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1],
                dim=0)

        elif color_mode == 'y':
            # Set mapping mask to point labels
            color = torch.ByteTensor(class_colors)[data.y]
            color = color.repeat_interleave(
                im.mappings.pointers[1:] - im.mappings.pointers[:-1],
                dim=0)
            color = color.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1],
                dim=0)
        
        elif color_mode == 'feat_3d':
            color = (feats_to_rgb(data.x) * 255).type(torch.uint8)
            color = color.repeat_interleave(
                im.mappings.pointers[1:] - im.mappings.pointers[:-1],
                dim=0)
            color = color.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1],
                dim=0)
        
        elif color_mode == 'feat_proj':
            color = (feats_to_rgb(im.mappings.features) * 255).type(torch.uint8)
            color = color.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1], 
                dim=0)

        # Apply the coloring to the mapping masks
        im.x[idx] = color

        # Apply a mask to the 3D pointwise error
        if show_point_error:
            is_tp = (data.pred == data.y)
            is_tp = is_tp.repeat_interleave(
                im.mappings.pointers[1:] - im.mappings.pointers[:-1],
                dim=0)
            is_tp = is_tp.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1],
                dim=0)
            is_tp = is_tp.unsqueeze(1)
            im.x[idx] = im.x[idx] * is_tp + ~is_tp * error_color

        # Apply a mask to the 3D view-wise error
        if show_view_error:
            y = data.y
            y = y.repeat_interleave(
                im.mappings.pointers[1:] - im.mappings.pointers[:-1],
                dim=0)
            y = y.repeat_interleave(
                im.mappings.values[1].pointers[1:]
                - im.mappings.values[1].pointers[:-1],
                dim=0)
            is_tp = (im.pred[idx] == y)
            is_tp = is_tp.unsqueeze(1)
            im.x[idx] = im.x[idx] * is_tp + ~is_tp * error_color

    # Prepare figure
    width = width if width and height else figsize
    height = height if width and height else int(figsize / 2)
    margin = int(0.02 * min(width, height))
    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(aspectmode='data', ),  # preserve aspect ratio
        margin=dict(l=margin, r=margin, b=margin, t=margin),)
    fig = go.Figure(layout=layout)
    fig.update_xaxes(visible=False)  # hide image axes
    fig.update_yaxes(visible=False)  # hide image axes

    # Draw the images
    n_img_traces = images.num_views
    for i, image in enumerate(chain(*[im.x.__iter__()
                                          for im in images])):
        fig.add_trace(
            go.Image(
                z=image.permute(1, 2, 0),
                visible=i == 0,  # initialize to image 0 visible
                hoverinfo='none',))  # disable hover info on images

    # Traces visibility for interactive point cloud coloring
    def trace_visibility(i_img):
        visibilities = np.array(
            [d.visible for d in fig.data], dtype='bool')
        if i_img < n_img_traces:
            visibilities[:] = False
            visibilities[i_img] = True
        return [{"visible": visibilities.tolist()}]

    # Create the buttons that will serve for toggling trace visibility
    updatemenus = [
        dict(
            buttons=[
                dict(label=f"{i_img}",
                     method='update',
                     args=trace_visibility(i_img))
                for i_img in range(n_img_traces)],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02,),]

    fig.update_layout(updatemenus=updatemenus)

    return fig


def figure_html(fig):
    # Save plotly figure to temp HTML
    fig.write_html(
        '/tmp/fig.html',
        config={'displayModeBar': False},
        include_plotlyjs='cdn',
        full_html=False)

    # Read the HTML
    with open("/tmp/fig.html", "r") as f:
        fig_html = f.read()

    # Center the figure div for cleaner display
    fig_html = fig_html.replace('class="plotly-graph-div" style="', 
        'class="plotly-graph-div" style="margin:0 auto;')

    return fig_html


def visualize_mm_data(
        mm_data, show_3d=True, show_2d=True, path=None, title=None, **kwargs):
    """Draw an interactive 3D visualization of the Data point cloud."""
    assert isinstance(mm_data, MMData)

    # Sanitize title and path
    if title is None:
        title = "Multimodal data"
    if path is not None:
        if osp.isdir(path):
            path = osp.join(path, f"{title}.html")
        else:
            path = osp.splitext(path)[0] + '.html'
        fig_html = f'<h1 style="text-align: center;">{title}</h1>'

    # Draw a figure for 3D data visualization
    if show_3d:
        fig_3d = visualize_3d(mm_data, **kwargs)
        if path is None:
            fig_3d.show(config={'displayModeBar': False})
        else:
            fig_html += figure_html(fig_3d)

    # Draw a figure for 2D data visualization
    if show_2d:
        fig_2d = visualize_2d(mm_data, **kwargs)
        if path is None:
            fig_2d.show(config={'displayModeBar': False})
        else:
            fig_html += figure_html(fig_2d)

    if path is not None:
        with open(path, "w") as f:
            f.write(fig_html)
    
    return
