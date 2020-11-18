from torch_points3d.datasets.multimodal.data import MMData
from torch_geometric.transforms import FixedPoints
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.core.data_transform.multimodal.projection import pose_to_rotation_matrix_numba
from torch_points3d.core.data_transform.multimodal import PointImagePixelMappingFromId
import plotly.graph_objects as go
import numpy as np
import torch


def rgb_to_plotly_rgb(rgb):
    """Convert torch.Tensor of float RGB values in [0, 1] to
    plotly-friendly RGB format.
    """
    assert isinstance(rgb, torch.Tensor) and rgb.max() <= 1.0 and rgb.dim() <= 2

    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)

    return [f"rgb{tuple(x)}" for x in (rgb * 255).int().numpy()]


def visualize_3d(mm_data, class_names=None, class_colors=None, class_opacities=None, figsize=800, width=None,
                 height=None, voxel=0.1, max_points=100000, pointsize=5, **kwargs):
    """3D data visualization with interaction tools."""
    assert isinstance(mm_data, MMData)

    # Subsample to limit the drawing time
    data_ = GridSampling3D(voxel)(mm_data.data.clone())
    if data_.num_nodes > max_points:
        data_ = FixedPoints(max_points, replace=False, allow_duplicates=False)(data_)
        
    # Subsample the mappings accordingly
    transform = PointImagePixelMappingFromId(key='point_index', keep_unseen_images=True)
    data_, images_, mappings_ = transform(data_, mm_data.images, mm_data.mappings)
    
    # Round to the cm for cleaner hover info
    data_.pos = (data_.pos * 100).round() / 100
    images_.pos = (images_.pos * 100).round() / 100
    
    # Prepare figure
    width = width if width and height else figsize
    height = height if width and height else int(figsize / 2)
    margin = int(0.02 * min(width, height))
    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(aspectmode='data', ),  # preserve aspect ratio
        margin=dict(l=margin, r=margin, b=margin, t=margin),
    )
    fig = go.Figure(layout=layout)

    # Draw a trace for RGB 3D point cloud
    fig.add_trace(
        go.Scatter3d(
            name='RGB',
            x=data_.pos[:, 0],
            y=data_.pos[:, 1],
            z=data_.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=rgb_to_plotly_rgb(data_.rgb),
            ),
            hoverinfo='x+y+z',
            showlegend=False,
            visible=True,
        )
    )
    n_rgb_traces = 1  # keep track of the number of traces

    # Draw a trace for labeled 3D point cloud
    y = data_.y.numpy()
    n_classes = int(y.max() + 1)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    if class_colors is None:
        class_colors = [None] * n_classes
    else:
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
                x=data_.pos[indices, 0],
                y=data_.pos[indices, 1],
                z=data_.pos[indices, 2],
                mode='markers',
                marker=dict(
                    size=pointsize,
                    color=class_colors[label],
                ),
                visible=False,
            )
        )
        n_y_traces += 1  # keep track of the number of traces

    # Draw a trace for 3D point cloud of number of images seen
    n_seen = torch.from_numpy(mappings_.jumps[1:] - mappings_.jumps[:-1])
    fig.add_trace(
        go.Scatter3d(
            name='Times seen',
            x=data_.pos[:, 0],
            y=data_.pos[:, 1],
            z=data_.pos[:, 2],
            mode='markers',
            marker=dict(
                size=pointsize,
                color=n_seen,
                colorscale='spectral',
                colorbar=dict(thickness=10, len=0.66, tick0=0, dtick=max(1, int(n_seen.max() / 10.)), ),
            ),
            hovertext=[f"seen: {n}" for n in n_seen],
            hoverinfo='x+y+z+text',
            showlegend=False,
            visible=False,
        )
    )
    n_seen_traces = 1  # keep track of the number of traces   

    # Draw image positions
    image_xyz = np.asarray(images_.pos.clone())
    image_opk = np.asarray(images_.opk.clone())
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
                        width=pointsize + 1
                    ),
                    showlegend=False,
                    hoverinfo='none',
                    visible=True,
                )
            )

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
                    size=pointsize + 4,
                ),
                text=f"<b>{i}</b>",
                textposition="bottom center",
                textfont=dict(
                    size=16
                ),
                hoverinfo='x+y+z+name',
                showlegend=False,
                visible=True,
            )
        )

    # Traces visibility for interactive point cloud coloring
    def trace_visibility(mode):
        visibilities = np.array([d.visible for d in fig.data], dtype='bool')

        # Traces visibility for interactive point cloud coloring
        if mode == 'rgb':
            visibilities[:n_rgb_traces + n_y_traces + n_seen_traces] = False
            visibilities[:n_rgb_traces] = True

        elif mode == 'labels':
            visibilities[:n_rgb_traces + n_y_traces + n_seen_traces] = False
            visibilities[n_rgb_traces:n_rgb_traces + n_y_traces] = True

        elif mode == 'n_seen':
            visibilities[:n_rgb_traces + n_y_traces + n_seen_traces] = False
            visibilities[n_rgb_traces + n_y_traces:n_rgb_traces + n_y_traces + n_seen_traces] = True

        return [{"visible": visibilities.tolist()}]

    # Create the buttons that will serve for toggling trace visibility
    updatemenus = [
        dict(
            buttons=[
                dict(label='RGB',
                     method='update',
                     args=trace_visibility('rgb')
                     ),
                dict(label='Labels',
                     method='update',
                     args=trace_visibility('labels')
                     ),
                dict(label='Times seen',
                     method='update',
                     args=trace_visibility('n_seen')
                     ),
            ],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02,
        ),
    ]
    fig.update_layout(updatemenus=updatemenus)

    # Place the legend on the left
    fig.update_layout(legend=dict(
        yanchor="middle",
        y=0.5,
        xanchor="right",
        x=0.99
    ))

    return fig


def visualize_2d(mm_data, image_batch=None, figsize=800, width=None, height=None, alpha=5, **kwargs):
    """2D data visualization with interaction tools."""
    assert isinstance(mm_data, MMData)

    # Read images to the resolution at which the mappings were computed
    if image_batch is None:
        image_batch = mm_data.images.read_images(size=mm_data.images.map_size_low)

    # Get the mapping of all points in the sample
    point_mapping = mm_data.mappings
    point_image_indices = torch.repeat_interleave(
        torch.from_numpy(point_mapping.values[0]),
        torch.from_numpy(point_mapping.values[1].jumps[1:] - point_mapping.values[1].jumps[:-1])
    )
    point_pixels = point_mapping.values[1].values[0]

    # Color the images where the point is projected
    color = torch.full((3,), alpha, dtype=torch.uint8)
    image_batch_ = image_batch.clone()
    image_batch_ = (image_batch_.float() / alpha).floor().type(torch.uint8)  # dark
    image_batch_[point_image_indices, :, point_pixels[:, 1], point_pixels[:, 0]] = \
        image_batch_[point_image_indices, :, point_pixels[:, 1], point_pixels[:, 0]] * color  # light

    # Prepare figure
    width = width if width and height else figsize
    height = height if width and height else int(figsize / 2)
    margin = int(0.02 * min(width, height))
    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(aspectmode='data', ),  # preserve aspect ratio
        margin=dict(l=margin, r=margin, b=margin, t=margin),
    )
    fig = go.Figure(layout=layout)
    fig.update_xaxes(visible=False)  # hide image axes
    fig.update_yaxes(visible=False)  # hide image axes

    # Draw the images
    n_img_traces = mm_data.images.num_images
    for i, image in enumerate(image_batch_):
        fig.add_trace(
            go.Image(
                z=image.permute(1, 2, 0),
                visible=i == 0,  # initialize to image 0 visible
                hoverinfo='none',  # disable hover info on images
            )
        )

    # Traces visibility for interactive point cloud coloring
    def trace_visibility(i_img):
        visibilities = np.array([d.visible for d in fig.data], dtype='bool')
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
                for i_img in range(n_img_traces)
            ],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02,
        ),
    ]
        
    fig.update_layout(updatemenus=updatemenus)

    return fig


def visualize_mm_data(mm_data, show_3d=True, show_2d=True, **kwargs):
    """Draw an interactive 3D visualization of the Data point cloud."""
    assert isinstance(mm_data, MMData)
    
    # Draw a figure for 3D data visualization
    if show_3d:
        fig_3d = visualize_3d(mm_data, **kwargs)
        fig_3d.show(config={'displayModeBar': False})
        
    # Draw a figure for 2D data visualization
    if show_2d:
        fig_2d = visualize_2d(mm_data, **kwargs)
        fig_2d.show(config={'displayModeBar': False})
    
    return
