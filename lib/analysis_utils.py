from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal


# Data manipulation functions
def old_parse_gpu_data(input_file: str) -> pd.DataFrame:
    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    df['time'] = df['timestamp'].dt.round('s')

    return df

def parse_gpu_data(input_file: str) -> pd.DataFrame:
    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    # Take mean over duplicated timestamps for each unique node, gpu id.
    df = df.groupby(['timestamp', 'node_id', 'gpu_id']).mean().reset_index()
    #select columns of interest
    pivot_df = df.pivot(index='timestamp', columns=['node_id', 'gpu_id'], values=['gract', 'smact', 'tenso', 'tmptr', 'power', 'pcitx', 'pcirx', 'nvltx', 'nvlrx', 'nvl0t', 'nvl0r', 'nvl1t', 'nvl1r', 'nvl2t', 'nvl2r', 'nvl3t', 'nvl3r'])
    pivot_df.reset_index(inplace=True)

    pivot_df.dropna(axis=1, how='all', inplace=True)

    pivot_df['time'] = pivot_df['timestamp'].dt.round('100ms') #not really necessary because data should already be in 100ms intervals.
    pivot_df.set_index('time', inplace=True)

    return old_parse_gpu_data(input_file), pivot_df

def parse_other_data(input_file: str) -> pd.DataFrame:
    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    return df

# Visualization functions
def old_plot_summary_series(df, y_col, title, y_label, include_std=False) -> go.Figure:
    """
    Plot a time series graph with mean and optional standard deviation.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        y_col (str): Column name for the y-axis.
        title (str): Title of the plot.
        y_label (str): Label for the y-axis.
        include_std (bool): Whether to include standard deviation in the plot.
    Returns:
        str: HTML representation of the plot.
    """

    default_color = px.colors.qualitative.Plotly[0]
    df_mean = df.mean()
    df_std = df.std()

    fig = px.line(
        df_mean,
        y=y_col,
        labels={y_col: y_label, 'time': 'Time'},
        title=title
    )

    if include_std:
        fig.add_trace(
            go.Scatter(
                x=df_mean.index,
                y=df_mean[y_col] - 2 * df_std[y_col],
                mode='lines',
                line={'width': 0, 'color': default_color},
                showlegend=False,
                name='2 Std Dev',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_mean.index,
                y=df_mean[y_col] + 2 * df_std[y_col],
                mode='lines',
                line={'width': 0, 'color': default_color},
                fill='tonexty',
                showlegend=False,
                name='2 Std Dev',
            )
        )

    return fig

def plot_summary_series(df, y_col, title, y_label, include_std=True,limit:int=5,smoothing_size:int=10, table=True) -> go.Figure:
    """
    Plot a time series graph with mean and optional standard deviation.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        y_col (str): Column name for the y-axis.
        title (str): Title of the plot.
        y_label (str): Label for the y-axis.
        include_std (bool): Whether to include standard deviation in the plot.
    Returns:
        str: HTML representation of the plot.
    """

    default_color = 'rgba(0.8,0.8,0.8,0.2)'

    #flatten_multilevel column
    df = df[y_col].copy()
    df.columns = ["_".join(a) for a in df.columns.to_flat_index()]

    mean = df.mean(axis=0) #Mean over time for fixed node and gpu
    df['mean'] = df.mean(axis=1) #Mean over nodes or gpu at fixed time
    df['std'] = df.std(axis=1)
    mean.sort_values(ascending=True, inplace=True)

    if include_std:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mean'] - 2 * df['std'],
                mode='lines',
                line={'width': 0, 'color': default_color},
                showlegend=False,
                name='2 Std Dev',
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['mean'] + 2 * df['std'],
                mode='lines',
                line={'width': 0, 'color': default_color},
                fill='tonexty',
                showlegend=False,
                name='2 Std Dev',
            )
        )
        #plots top limit and bottom limit of gpus
        for x in mean.index[-limit:].insert(0,mean.index[:limit]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    # y=df[x].rolling(smoothing_size).mean()
                    y=signal.savgol_filter(df[x].dropna(),#drop na otherwise the filter fails
                               smoothing_size,  # window size used for filtering
                               1),#maybe 3 is good too
                    mode='lines',
                    line={'width': 0.5},
                    showlegend=True,
                    name=f"{x.split('_')[0]}-GPU{x.split('_')[1]}",
                )
            )

    else:
        fig = px.line(
            df['mean'],
            y=y_col,
            labels={y_col: y_label, 'time': 'Time'},
            title=title
        )

    fig.update_traces(line={'width': 0.5})
    return fig


def generate_plotly_table(df: pd.DataFrame) -> go.Figure:
    """
    Generate an interactive HTML table using Plotly.

    Args:
        df (pd.DataFrame): DataFrame to be converted into a table.

    Returns:
        str: HTML string of the table.
    """
    return go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color='lightgrey',
                    align='left',
                    font=dict(size=12, color='black'),
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color='white',
                    align='left',
                    font=dict(size=12, color='black'),
                )
            )
        ]
    )


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


def get_key_statistics(gpu_data: pd.DataFrame, anomalies: int) -> List[List]:
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

    return [
        [
            'Total nodes',
            gpu_data['node_id'].nunique(),
            'primary'
        ], [
            'Total GPUs',
            gpu_data.groupby(['node_id', 'gpu_id']).ngroups,
            'primary'
        ], [
            'Anomalies detected',
            anomalies,
            'warning'
        ], [
            'Reporting Duration (min)',
            f'{minutes}:{seconds:02d}',  # Format as MM:SS
            'primary'
        ],
    ]

def get_overview_statistics(gpu_data: pd.DataFrame) -> List[go.Figure]:
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
    power_vs_tenso = px.scatter(
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

    return [power_vs_tenso]

def old_get_temp_statistics(gpu_data: pd.DataFrame) -> List[go.Figure]:
    """
    Get the temperature statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """

    gpu_data['time'] = gpu_data['timestamp'].dt.round('s')
    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]

    power_timeline = plot_summary_series(
        time_data,
        'tmptr',
        'Mean Temperature Over Time',
        'Temperature (°C)',
        include_std=True,
    )
    plots = [power_timeline]

    temp_outliers = find_series_outliers(gpu_data, 'tmptr')

    if len(temp_outliers) > 0:
        top_hot_gpus = temp_outliers.head(20)
        temp_outliers_bar = px.bar(
            top_hot_gpus,
            x='count',
            y=[f'{row['node_id']}-{row['gpu_id']}' for _, row in top_hot_gpus.iterrows()],
            labels={'count': 'Count of High Temperature Readings', 'y': 'GPU'},
            title='GPUs with Frequent High Temperatures'
        )
        temp_outliers_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        plots.append(temp_outliers_bar)

        # Plot temperature for GPUs with most outliers
        temp_outliers_fig = go.Figure()
        for _, row in temp_outliers.head(5).iterrows():
            gpu_id = row['gpu_id']
            node = row['node_id']
            gpu_temp_data = gpu_data[(gpu_data['gpu_id'] == gpu_id) & (gpu_data['node_id'] == node)]
            temp_outliers_fig.add_trace(
                go.Scatter(
                    x=gpu_temp_data['time'],
                    y=gpu_temp_data['tmptr'],
                    opacity=0.7,
                    name=f'{node}-{gpu_id}'
                )
            )

        temp_outliers_fig.update_layout(
            title='GPUs with Most Temperature Outliers',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
        )

    return plots

def get_temp_statistics(pivot_gpu_data: pd.DataFrame) -> List[go.Figure]:
    """
    Get the temperature statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    plots = []
    temp_timeline = plot_summary_series(
        pivot_gpu_data,
        'tmptr',
        'Mean Temperature Over Time',
        'Temperature (°C)',
        include_std=True,
    )
    table1 = anomalies_table(pivot_gpu_data, 'tmptr', True, 5)
    table2 = anomalies_table(pivot_gpu_data, 'tmptr', False, 5)

    pivot_gpu_data = pivot_gpu_data['tmptr'].copy()
    pivot_gpu_data.columns = ["_".join(a) for a in pivot_gpu_data.columns.to_flat_index()]

    mean = pivot_gpu_data.mean(axis=0) #Mean over time for fixed node and gpu
    mean.sort_values(ascending=True, inplace=True)

    # temp distribution
    temp_distribution = px.histogram(
        x=mean,
        labels={'temperature': 'Degrees Celsius', 'count': 'Count'},
        title='Mean Temperature of GPUs',
    )

    plots.append(temp_timeline)
    plots.append(temp_distribution)
    plots.append(table1)
    plots.append(table2)

    return plots

def old_get_power_statistics(gpu_data: pd.DataFrame) -> List[go.Figure]:
    """
    Get the power statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """

    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]

    # Power usage plots
    plots = []
    plots.append(plot_summary_series(
        time_data,
        'power',
        'Mean Power Over Time',
        'Power (W)',
        include_std=True,
    ))

    # Power distribution
    power_distribution = px.histogram(
        gpu_data.groupby(['node_id', 'gpu_id']).agg({'power': 'mean'}),
        x='power',
        labels={'power': 'Power (W)', 'count': 'Count'},
        title='Mean GPU Power Usage',
    )
    plots.append(power_distribution)

    # Identify GPUs with high power usage
    power_outliers = find_series_outliers(gpu_data, 'power')

    if len(power_outliers) > 0:
        # Plot Power for GPUs with most outliers
        power_outliers_fig = go.Figure()

        for _, row in power_outliers.head(5).iterrows():
            gpu_id = row['gpu_id']
            node = row['node_id']
            gpu_power_data = gpu_data[(gpu_data['gpu_id'] == gpu_id) & (gpu_data['node_id'] == node)]
            power_outliers_fig.add_trace(
                go.Scatter(
                    x=gpu_power_data['time'],
                    y=gpu_power_data['power'],
                    opacity=0.7,
                    name=f'{node}-{gpu_id}'
                )
            )

        power_outliers_fig.update_layout(
            title='GPUs with Most Power Outliers',
            xaxis_title='Time',
            yaxis_title='Power (W)',
        )

        plots.append(power_outliers_fig)

    return plots

def get_power_statistics(pivot_gpu_data: pd.DataFrame) -> List[go.Figure]:
    """
    Get the power statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """

    # Power usage plots
    plots = []
    plots.append(plot_summary_series(
        pivot_gpu_data,
        'power',
        'Mean Power Over Time',
        'Power (W)',
        include_std=True,
    ))

    table1 = anomalies_table(pivot_gpu_data, 'power', True, 5)
    table2 = anomalies_table(pivot_gpu_data, 'power', True, 5)

    pivot_gpu_data = pivot_gpu_data['tmptr'].copy()
    pivot_gpu_data.columns = ["_".join(a) for a in pivot_gpu_data.columns.to_flat_index()]

    mean = pivot_gpu_data.mean(axis=0) #Mean over time for fixed node and gpu
    mean.sort_values(ascending=True, inplace=True)

    # power distribution
    power_distribution = px.histogram(
        x=mean,
        labels={'temperature': 'Degrees Celsius', 'count': 'Count'},
        title='Mean Temperature of GPUs',
    )

    plots.append(power_distribution)
    plots.append(table1)
    plots.append(table2)

    return plots

def get_activity_statistics(gpu_data: pd.DataFrame) -> List[go.Figure]:
    """
    Get the activity statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        list: list of plots/tables/comments that need to be displayed
    """
    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]
    plots = []

    time_avg = time_data.mean().reset_index()
    activity_timeline = make_subplots(specs=[[{'secondary_y': True}]])
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
        y=signal.savgol_filter(time_avg['tenso'],
                               5,  # window size used for filtering
                               2),  # order of fitted polynomial
        name='Smoothed Tensor Core Activity',
    ),
        secondary_y=True,
    )
    activity_timeline.update_layout(title='GPU Activity Over Time')
    activity_timeline.update_xaxes(title_text='Time')
    activity_timeline.update_yaxes(title_text='Tensor Core Usage', secondary_y=False)
    activity_timeline.update_yaxes(title_text='Graphic Activity', secondary_y=True)
    plots.append(activity_timeline)

    return plots

def find_anomalies(gpu_data: pd.DataFrame) -> pd.DataFrame:
    """
    Get the anomaly statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        pd.DataFrame: DataFrame containing the anomalies.
    """

    # Power efficiency (tensor operations per watt)
    gpu_data['efficiency'] = gpu_data['tenso'] / (gpu_data['power'] + 1e-5)

    # Usage balance (ratio of SM active to tensor operations)
    gpu_data['usage_balance'] = gpu_data['smact'] / (gpu_data['tenso'] + 1e-5)

    node_avg = gpu_data.groupby(['node_id', 'gpu_id']).agg({
        'gract': 'mean',
        'smact': 'mean',
        'tenso': 'mean',
        'tmptr': 'mean',
        'power': 'mean'
    }).reset_index()

    anomalies = []
    anomalies_types = {
        'gract': 'Graphic Activity',
        'tmptr': 'Temperature',
        'power': 'Power',
    }

    for col, title in anomalies_types.items():
        outliers = find_series_outliers(gpu_data, col)
        outliers['anomaly'] = title
        anomalies.append(outliers)

    # Combine the anomalies in a unique dataframe
    anomalies = pd.concat(anomalies, ignore_index=True)
    # Merge with the node_avg dataframe
    anomalies = pd.merge(anomalies, node_avg, on=['node_id', 'gpu_id'], how='left')
    # Group by Node and GPU, concat the 'anomaly' fields
    return anomalies.groupby(['node_id', 'gpu_id']).agg({
        'count': 'sum',
        'anomaly': lambda x: ', '.join(x)
    }).sort_values('count', ascending=False).reset_index()

def anomalies_table(pivot_gpu_data: pd.DataFrame, metric:str, asc = True, limit:int = 5) -> go.Figure:
    pivot_gpu_data = pivot_gpu_data[metric].copy()
    pivot_gpu_data.columns = ["_".join(a) for a in pivot_gpu_data.columns.to_flat_index()]

    mean = pivot_gpu_data.mean(axis=0) #Mean over time for fixed node and gpu
    mean.sort_values(ascending=asc, inplace=True)
    fig = go.Figure()
    mean_df = mean.reset_index().iloc[:limit]

    fig.add_trace(
        go.Table(
            name = 'Lowest Outlier' if asc else 'Highest Outlier',
            header=dict(
                values=['gpu', 'mean'],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[mean_df[k].tolist() for k in mean_df.columns],
                align="left")
        ),
    )
    fig.update_layout(
        title=dict(
            text='Lowest Outlier' if asc else 'Highest Outlier'
        )
    )
    return fig