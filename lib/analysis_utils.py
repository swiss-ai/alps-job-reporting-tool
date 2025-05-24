import math
from typing import List, Tuple, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

type Renderable = Union[go.Figure, pd.DataFrame, str]

# Data manipulation functions
def parse_gpu_data(input_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    df['time'] = df['timestamp'].dt.round('s')

    # Take mean over duplicated timestamps for each unique node, gpu id.
    df = df.groupby(['timestamp', 'node_id', 'gpu_id']).mean().reset_index()

    # Select columns of interest
    pivot_metrics = ['drama', 'gputl', 'fbusp', 'mcutl', 'mmtmp', 'gract',
                     'smact', 'tenso', 'tmptr', 'power', 'pcitx', 'pcirx',
                     'nvltx', 'nvlrx', 'nvl0t', 'nvl0r', 'nvl1t', 'nvl1r',
                     'nvl2t', 'nvl2r', 'nvl3t', 'nvl3r']
    pivot_df = df.pivot(index='timestamp', columns=['node_id', 'gpu_id'], values=pivot_metrics)
    pivot_df.reset_index(inplace=True)
    pivot_df.dropna(axis=1, how='all', inplace=True)

    # Rounding is not really necessary because data should already be in 100 ms intervals.
    pivot_df['time'] = pivot_df['timestamp'].dt.round('100ms')
    pivot_df.set_index('time', inplace=True)

    return df, pivot_df


def parse_other_data(input_file: str) -> pd.DataFrame:
    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Remove spaces from column names
    df.columns = df.columns.str.replace(' ', '')

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    return df


# Visualization functions
def plot_summary_series(df, y_col, title, y_label, include_std=True, limit: int = 5, smoothing: int = 10) -> go.Figure:
    """
    Plot a time series graph with mean and optional standard deviation.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        y_col (str): Column name for the y-axis.
        title (str): Title of the plot.
        y_label (str): Label for the y-axis.
        include_std (bool): Whether to include standard deviation in the plot.
        limit (int): The maximum number of components to plot.
        smoothing (int): The size of the smoothing window.
    Returns:
        str: HTML representation of the plot.
    """

    node_mean = df[y_col].mean(axis=0)  # Mean for a fixed component
    time_mean = df[y_col].mean(axis=1)  # Mean for a fixed time

    if include_std:
        time_std = df[y_col].std(axis=1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=time_mean - 2 * time_std,
                mode='lines',
                line=dict(width=0, color='rgba(0.8,0.8,0.8,0.2)'),
                showlegend=False,
                name='2 Std Dev',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=time_mean + 2 * time_std,
                mode='lines',
                line=dict(width=0, color='rgba(0.8,0.8,0.8,0.2)'),
                fill='tonexty',
                showlegend=False,
                name='2 Std Dev',
            )
        )

        largest = node_mean.nlargest(limit).index
        smallest = node_mean.nsmallest(limit).index

        # Plots top and bottom limits
        for x in smallest.union(largest).values:
            smoothed = signal.savgol_filter(df[y_col][x].dropna(), smoothing, 1)

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=smoothed,
                    mode='lines',
                    showlegend=True,
                    name=x if x is str else f'{x[0]}-GPU{x[1]}',
                )
            )

    else:
        fig = px.line(
            time_mean,
            y=y_col,
            labels=dict(y_col=y_label, time='Time'),
            title=title,
        )

    fig.update_traces(line=dict(width=0.75))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title=y_label)
    return fig


# Data analysis functions
def find_series_outliers(df: pd.DataFrame, y_col: str, n_std=3) -> pd.DataFrame:
    """
    Find outliers in a time series based on the mean and standard deviation.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        y_col (str): Column name for the y-axis.
        n_std (int): Number of standard deviations to consider an outlier.
    Returns:
        pd.DataFrame: DataFrame containing the outliers.
    """
    outliers = df[df[y_col] > df[y_col].mean() + n_std * df[y_col].std()]

    return outliers.groupby(['node_id', 'gpu_id']).size() \
        .reset_index(name='count') \
        .sort_values('count', ascending=False)


def get_key_statistics(gpu_data: pd.DataFrame, other_data: pd.DataFrame) -> List[List]:
    """
    Get the key statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data.
        anomalies (int): The number of anomalies detected.

    Returns:
        list: [title, value, type] for each key statistic.
    """
    duration = gpu_data['timestamp'].max() - gpu_data['timestamp'].min()
    minutes, seconds = divmod(int(duration.total_seconds()), 60)
    data_points = gpu_data.count().sum() + other_data.count().sum()
    # Format number of data points as human-readable format
    magnitude = min(int(math.log10(data_points)) // 3, 4)
    suffix = ['', 'K', 'M', 'B', 'T'][magnitude]

    return [
        [
            'Total Nodes',
            gpu_data['node_id'].nunique(),
            'primary'
        ], [
            'Total GPUs',
            gpu_data.groupby(['node_id', 'gpu_id']).ngroups,
            'primary'
        ], [
            'Data Points',
            f'{data_points / 10 ** (magnitude * 3):.2f}{suffix}',
            'primary'
        ], [
            'Reporting Duration (min)',
            f'{minutes}:{seconds:02d}',  # Format as MM:SS
            'primary'
        ],
    ]


def gpu_overview_statistics(gpu_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the overview statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    # Prepare data for overview stats
    node_avg = gpu_data.groupby(['node_id', 'gpu_id']).agg({
        'gract': 'mean',
        'smact': 'mean',
        'tenso': 'mean',
        'tmptr': 'mean',
        'power': 'mean'
    }).reset_index()

    # Power vs Tensor Usage plot
    power_vs_tenso_plot = px.scatter(
        node_avg,
        x='tenso',
        y='power',
        color='tmptr',
        labels={
            'tenso': 'Tensor Core Usage',
            'power': 'Power Consumption (W)',
            'tmptr': 'Temp (°C)',
            'node_id': 'Node ID',
            'gpu_id': 'GPU ID',
        },
        hover_data=['node_id', 'gpu_id'],
        title='Power vs. Tensor Usage',
        color_continuous_scale='Viridis',
    )

    # List individual GPU data
    columns = {
        'node_id': 'Node ID',
        'gpu_id': 'GPU ID',
        'power': 'Power (W)',
        'tmptr': 'Device Temp (°C)',
        'mmtmp': 'Memory Temp (°C)',
    }
    gpu_table = gpu_data.groupby(['node_id', 'gpu_id']).mean() \
        .reset_index()[columns.keys()].rename(columns=columns)

    # Create correlation matrix
    corr_metrics = ['gract', 'smact', 'power', 'tmptr', 'pcitx', 'pcirx', 'nvltx', 'nvlrx']
    corr_metrics_labels = [s.upper() for s in corr_metrics]
    corr_plot = px.imshow(
        gpu_data[corr_metrics].corr().round(2),
        labels=dict(color='Coolwarm'),
        x=corr_metrics_labels,
        y=corr_metrics_labels,
        text_auto=True,
        aspect='auto',
        title='Metrics Correlation Matrix',
    )
    corr_plot.update_xaxes(side='top')

    return [power_vs_tenso_plot, corr_plot, gpu_table]


def gpu_temp_statistics(pivot_gpu_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the temperature statistics from the GPU data.

    Args:
        pivot_gpu_data (pd.DataFrame): The GPU data, pivoted DataFrame.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    plots: List[Renderable] = [
        'Plot shows the 2 standard deviation interval and the 5 gpus with the lowest and highest mean temperatures over time. Savitzky-Golay filter has been applied to the lines plotted with window size = 10 and polynomial order = 1. Missing values have been dropped and the smallest time interval between any two data points is 100ms',
    ]
    metrics = {
        'tmptr': 'Device Temperature',
        'mmtmp': 'Memory Temperature',
    }

    for metric, name in metrics.items():
        temp_timeline = plot_summary_series(
            pivot_gpu_data,
            y_col=metric,
            title=name,
            y_label='Temperature (°C)',
            include_std=True,
        )
        plots.append(temp_timeline)

    metrics_mean = pivot_gpu_data[['tmptr', 'mmtmp']].mean(axis=0) \
        .rename(dict(tmptr='Device Temp', mmtmp='Memory Temp'))

    temp_distribution = px.histogram(
        x=metrics_mean,
        color=metrics_mean.index.get_level_values(0),
        labels=dict(x='Temperature (°C)', color='Type'),
        title='Mean Temperature Distribution',
        opacity=0.7,
    )
    temp_distribution.update_layout(yaxis_title='Count')
    plots.append(temp_distribution)

    return plots


def gpu_power_statistics(gpu_pivot_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the power statistics from the GPU data.

    Args:
        gpu_pivot_data (pd.DataFrame): The GPU data, pivoted DataFrame.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """

    # Power usage plots
    plots: List[Renderable] = [
        'Plot shows the 2 standard deviation interval and the 5 gpus with the lowest and highest mean power usage over time. Savitzky-Golay filter has been applied to the lines plotted with window size = 10 and polynomial order = 1. Missing values have been dropped and the smallest time interval between any two data points is 100ms',
        plot_summary_series(
            gpu_pivot_data,
            'power',
            'Power Consumption',
            'Power (W)',
            include_std=True,
        )
    ]

    # Power distribution plot
    power_distribution = px.histogram(
        x=gpu_pivot_data['power'].mean(axis=0),
        labels=dict(x='Power (W)'),
        title='Mean Power Distribution',
    )
    power_distribution.update_layout(yaxis_title='Count')
    plots.append(power_distribution)

    return plots


def get_activity_statistics(gpu_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the activity statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]
    time_avg = time_data.mean().reset_index()

    smoothed_tenso = signal.savgol_filter(time_avg['tenso'], 5, 2)  # window size and polynomial order

    activity_timeline = make_subplots(specs=[[dict(secondary_y=True)]])
    activity_timeline.add_trace(
        go.Scatter(
            x=time_avg['time'],
            y=time_avg['gract'],
            name='Graphic Activity',
        ),
        secondary_y=False,
    )
    activity_timeline.add_trace(
        go.Scatter(
            x=time_avg['time'],
            y=time_avg['tenso'],
            name='Tensor Core Activity',
        ),
        secondary_y=True,
    )
    activity_timeline.add_trace(
        go.Scatter(
            x=time_avg['time'],
            y=smoothed_tenso,
            name='Smoothed Tensor Core Activity',
        ),
        secondary_y=True,
    )
    activity_timeline.update_layout(title='GPU Activity Over Time', xaxis_title='Time')
    activity_timeline.update_yaxes(title_text='Tensor Core Usage', secondary_y=False)
    activity_timeline.update_yaxes(title_text='Graphic Activity', secondary_y=True)

    return [activity_timeline]


def gpu_utilization_statistics(pivot_gpu_data: pd.DataFrame, limit: int = 5) -> List[Renderable]:
    """
    Get the utilization statistics from the GPU data.
    Args:
        pivot_gpu_data (pd.DataFrame): The GPU data, pivoted DataFrame.
        limit (int): The number of least used GPUs to display.
    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    plots: List[Renderable] = [
        'Least utilised gpus selected. Savitzky-Golay filter has been applied to the lines plotted with window size = 20 and polynomial order = 1. Missing values have been dropped and the smallest time interval between any two data points is 100ms.',
    ]
    metrics = {
        'drama': 'Device Memory Interface Utilization',
        'gputl': 'GPU Utilization',
        'mcutl': 'Memory Controller Utilization',
        'gract': 'Graphics Engine Utilization',
        'smact': 'SM Utilization',
        'tenso': 'Memory Utilization',
    }

    for metric, name in metrics.items():
        nodes = (pivot_gpu_data[metric] - 1).sum().sort_values(ascending=True).index[:limit]

        plot = make_subplots()

        for node in nodes:
            smoothed = signal.savgol_filter(pivot_gpu_data[metric].T.loc[node].dropna(), 20, 1)

            plot.add_trace(
                go.Scatter(
                    x=pivot_gpu_data.index,
                    y=smoothed,
                    name=f"{node[0]}-GPU{node[1]}",
                    mode='lines',
                    opacity=0.5,
                    marker=dict(
                        size=5,
                        line=dict(width=0.5),
                    )
                ),
            )

        plot.update_layout(title=name, xaxis_title='Time', yaxis_title='Utilization')
        plots.append(plot)

    return plots


def gpu_nvlink_statistics(pivot_gpu_data: pd.DataFrame, limit: int = 3) -> List[Renderable]:
    """
    Get the NVLink statistics from the GPU data.
    Args:
        pivot_gpu_data (pd.DataFrame): The GPU data, pivoted DataFrame.
        limit (int): The number of least used NVLinks to display.
    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    plots: List[Renderable] = [
        'Highest error counts in sum of both T&R selected. Savitzky-Golay filter has been applied to the lines plotted with window size = 100 and polynomial order = 1. Missing values have been dropped and the smallest time interval between any two data points is 100ms.',
    ]

    for i in range(4):
        metrics = [f'nvl{i}t', f'nvl{i}r']
        nodes = pivot_gpu_data[metrics].T.groupby(level=(1, 2)) \
                    .sum().sum(1).sort_values(ascending=False).iloc[:limit].index

        plot = make_subplots()

        for node in nodes:
            for metric in metrics:
                smoothed = signal.savgol_filter(pivot_gpu_data[metric].T.loc[node].dropna(), 100, 1)

                plot.add_trace(
                    go.Scatter(
                        x=pivot_gpu_data.index,
                        y=smoothed,
                        name=f'{node[0]}-GPU{node[1]} {metric[-1].upper()}',
                        mode='lines',
                        opacity=0.5,
                        marker=dict(
                            size=5,
                            line=dict(width=0.5)
                        )
                    ),
                )

        plot.update_layout(
            title=f'NVL{i}T/R',
            xaxis_title='Time',
            yaxis_title='Error Count',
        )
        plots.append(plot)

    return plots


def analyze_net_outliers(mean_std_df: pd.DataFrame, new_net_df: pd.DataFrame, column: str) -> go.Figure:
    """
    Analyzes outliers for a given column based on mean ± 2*std and generates a Plotly figure.

    Parameters:
        mean_std_df (pd.DataFrame): DataFrame containing mean and std for each timestamp.
        new_net_df (pd.DataFrame): Original DataFrame with node-level data.
        column (str): The column to analyze (e.g., 'rx_bytes').

    Returns:
        go.Figure: A Plotly figure visualizing the outliers.
    """
    # Merge the mean and std values with the original DataFrame
    specific_df = pd.merge(
        new_net_df[['timestamp', 'node_id', column]],
        mean_std_df[['timestamp', f'{column}_mean', f'{column}_std']],
        on='timestamp'
    )

    # Identify outliers (values greater or lower than mean +- 2*std)
    outlier_df = specific_df[
        (specific_df[column] > (specific_df[f'{column}_mean'] + 2 * specific_df[f'{column}_std'])) |
        (specific_df[column] < (specific_df[f'{column}_mean'] - 2 * specific_df[f'{column}_std']))
    ]

    # Group by node_id and count the number of outliers
    outliers = outlier_df.groupby('node_id').size().reset_index(name='outliers')

    # Filter nodes with outliers greater than 10% of the total timestamps
    duration = (new_net_df['timestamp'].max() - new_net_df['timestamp'].min()).total_seconds()
    outliers = outliers[outliers['outliers'] > (0.2 * duration)]
    outliers_node = outliers['node_id'].tolist()

    # Create a Plotly figure for the outliers
    fig = go.Figure()

    # Add a line for the mean
    fig.add_trace(
        go.Scatter(
            x=specific_df['timestamp'],
            y=specific_df[f'{column}_mean'],
            mode='lines',
            line=dict(dash='dash'),
            name='Mean',
        )
    )

    # Add traces for each outlier node
    for node in outliers_node:
        node_df = specific_df[specific_df['node_id'] == node]
        fig.add_trace(
            go.Scatter(
                x=node_df['timestamp'],
                y=node_df[column],
                mode='lines',
                name=f'Node {node}'
            )
        )

    # Add the mean ± 2*std range as a shaded area
    fig.add_trace(
        go.Scatter(
            x=specific_df['timestamp'],
            y=specific_df[f'{column}_mean'] + 2 * specific_df[f'{column}_std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Mean + 2*Std',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=specific_df['timestamp'],
            y=specific_df[f'{column}_mean'] - 2 * specific_df[f'{column}_std'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(width=0),
            showlegend=False,
            name='Mean - 2*Std',
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{column.replace('_', ' ').title()} Outliers",
        xaxis_title="Timestamp",
        yaxis_title=column.replace('_', ' ').title(),
        legend_title="Nodes",
    )

    return fig, outliers_node


def net_statistics(other_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the network statistics from the GPU data.

    Args:
        other_data (pd.DataFrame): The other data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    plots = []
    # prepare net_df
    net_columns = ['timestamp', 'node_id', 'hsn0_rx_bytes', 'hsn1_rx_bytes', 'hsn2_rx_bytes', 'hsn3_rx_bytes', 'nmn0_rx_bytes',
        'hsn0_rx_errors', 'hsn1_rx_errors', 'hsn2_rx_errors', 'hsn3_rx_errors',
        'nmn0_rx_errors', 'hsn0_rx_packets', 'hsn1_rx_packets',
        'hsn2_rx_packets', 'hsn3_rx_packets', 'nmn0_rx_packets',
        'hsn0_tx_bytes', 'hsn1_tx_bytes', 'hsn2_tx_bytes', 'hsn3_tx_bytes',
        'nmn0_tx_bytes', 'hsn0_tx_errors', 'hsn1_tx_errors', 'hsn2_tx_errors',
        'hsn3_tx_errors', 'nmn0_tx_errors', 'hsn0_tx_packets',
        'hsn1_tx_packets', 'hsn2_tx_packets', 'hsn3_tx_packets',
        'nmn0_tx_packets']
    net_df = other_data[net_columns]

    # prepare the 2 dataframes needed for the analysis
    new_columns = ['timestamp', 'node_id', 'rx_bytes', 'rx_errors', 'rx_packets', 'tx_bytes', 'tx_errors', 'tx_packets']
    # create a new dataframe with the new columns (created with a mean of the old columns)
    new_net_df = pd.DataFrame(columns=new_columns)
    # copy the timestamp and node_id columns from the old dataframe
    new_net_df['timestamp'] = net_df['timestamp']
    new_net_df['node_id'] = net_df['node_id']
    # for every column in the new dataframe combine the columns in the old dataframe
    for column in new_columns[2:]:
        # get the columns that start with the same prefix
        cols = [col for col in net_df.columns if col.endswith(column)]
        # get the mean of the columns
        new_net_df[column] = net_df[cols].sum(axis=1)

    # create a new dataframe with only mean and std for every column (grouped by timestamp)
    mean_std_df = new_net_df.drop(columns=['node_id']).groupby("timestamp").agg(['mean', 'std']).reset_index()
    # Flatten the MultiIndex columns in mean_std_df
    mean_std_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in mean_std_df.columns]

    # call the function to analyze the outliers for rx_bytes
    rx_bytes_fig, rx_bytes_outliers = analyze_net_outliers(mean_std_df, new_net_df, 'rx_bytes')
    tx_bytes_fig, tx_bytes_outliers = analyze_net_outliers(mean_std_df, new_net_df, 'tx_bytes')
    #print(rx_bytes_outliers)

    # add to plots
    plots.append(rx_bytes_fig)
    plots.append(tx_bytes_fig)

    return plots


def io_statistics(other_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the I/O statistics from the data.

    Args:
        other_data (pd.DataFrame): The input data containing I/O metrics.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    plots: List[Renderable] = [
        'In the following plots, we will se for every metric some plots to better understand the data. The first plot is a timeline of the metric over time, the second plot is a histogram of the mean values for each node, and the third plot is a table with the mean values for each node.',
    ]
    metrics = {
        'rchar': 'Bytes Read',
        'wchar': 'Bytes Written',
        'syscr': 'System Calls Read',
        'syscw': 'System Calls Write',
        'read_bytes': 'Bytes Read from Disk',
        'write_bytes': 'Bytes Written to Disk',
        'cancelled_write_bytes': 'Cancelled Write Bytes',
    }

    for metric, name in metrics.items():
        # Pivot the DataFrame to reshape it
        df = other_data.pivot(index='timestamp', columns='node_id', values=metric).dropna()

        # Plot the timeline for the metric
        timeline = plot_summary_series(
            df=df,
            y_col=df.columns,
            title=name,
            y_label=name,
            include_std=True,
        )

        # Create a histogram for the distribution of the mean values
        distribution = px.histogram(
            x=df.mean(axis=0),
            labels=dict(x=name),
            title=f'Mean {name} Distribution',
        )

        # Add the plots to the list
        plots.append(timeline)
        plots.append(distribution)

    columns = {
        'node_id': 'Node',
        'rchar': 'Read B',
        'wchar': 'Written B',
        'syscr': 'Read Sys Calls',
        'syscw': 'Write Sys Calls',
        'read_bytes': 'Disk Read B',
        'write_bytes': 'Disk Write B',
        'cancelled_write_bytes': 'Cancelled W. B',
    }
    io_table = other_data.groupby('node_id').mean() \
        .reset_index()[columns.keys()].rename(columns=columns)
    plots.append(io_table)

    return plots


def cpu_overview_statistics(other_data: pd.DataFrame) -> List[Renderable]:
    """
    Get the overview of CPU statistics from the data.
    Args:
        other_data (pd.DataFrame): The input data containing CPU metrics.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    columns = {
        'node_id': 'Node',
        'curr_mean': 'Curr. Mean',
        'curr_min': 'Curr. Min',
        'curr_max': 'Curr. Max',
        'power_mean': 'Power Mean',
        'power_min': 'Power Min',
        'power_max': 'Power Max',
        'temp_mean': 'Temp. Mean',
        'temp_min': 'Temp. Min',
        'temp_max': 'Temp. Max',
    }
    nodes_table = other_data.groupby('node_id').mean().reset_index()[columns.keys()] \
        .rename(columns=columns)

    return [nodes_table]


def cpu_statistics(other_data: pd.DataFrame, metric: str, name: str, unit: str) -> List[Renderable]:
    """
    Get the temperature, current, power statistics from the CPU data.

    Args:
        other_data (dict): The CPU data, raw data.
        metric (str): The key for the metric.
        name (str): The name of the metric.
        unit (str): The unit of the metric.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    aggregates = ['max', 'mean', 'min']
    metrics = {agg: f'{metric}_{agg}' for agg in aggregates}
    df = other_data.pivot(index='timestamp', columns=['node_id'], values=metrics.values())

    plots: List[Renderable] = [
        'Plot shows the 2 standard deviation interval and the 5 cpus with the lowest and highest mean measurements over time. Savitzky-Golay filter has been applied to the lines plotted with window size = 10 and polynomial order = 1. Missing values have been dropped and the smallest time interval between any two data points is 1s',
    ]

    for metric, col in metrics.items():
        timeline = plot_summary_series(
            df,
            y_col=col,
            title=f'{metric.title()} {name}',
            y_label=f'{name} ({unit})',
            include_std=True,
        )

        distribution = px.histogram(
            x=df[col].mean(axis=0),
            labels=dict(x=unit),
            title=f'Distribution of {metric.title()} {name}',
        )
        distribution.update_layout(yaxis_title_text='Count')

        plots.append(timeline)
        plots.append(distribution)

    return plots
