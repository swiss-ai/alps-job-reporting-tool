import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# data manipulation functions
def parse_gpu_data(input_file):
    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    return df


# visualization functions
def fig_to_html(fig):
    return fig.to_html(include_plotlyjs=False, full_html=False)


def plot_time_series(df, y_col, title, y_label, include_std=False):
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

    return fig_to_html(fig)


def generate_plotly_table(dataframe):
    """
    Generate an interactive HTML table using Plotly.

    Args:
        dataframe (pd.DataFrame): DataFrame to be converted into a table.

    Returns:
        str: HTML string of the table.
    """
    # Create Plotly Table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(dataframe.columns),
                    fill_color='lightgrey',
                    align='left',
                    font=dict(size=12, color='black'),
                ),
                cells=dict(
                    values=[dataframe[col] for col in dataframe.columns],
                    fill_color='white',
                    align='left',
                    font=dict(size=12, color='black'),
                )
            )
        ]
    )
    # Return the HTML of the table
    return fig_to_html(fig)


# data analysis functions
def get_key_statistics(gpu_data):
    """
    Get the key statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data.

    Returns:
        list: [title, value, type] for each key statistic.
    """

    stats = []

    # Total nodes
    stats.append([
        "Total nodes",
        gpu_data['node_id'].nunique(),
        "primary"
    ])

    # Total GPUs
    stats.append([
        "Total GPUs",
        gpu_data.groupby(['node_id', 'gpu_id']).ngroups,
        "primary"
    ])

    # Anomalies detected
    stats.append([
        "Anomalies detected",
        "TO-DO",
        "warning"
    ])

    # Reporting Duration
    duration = gpu_data['timestamp'].max() - gpu_data['timestamp'].min()
    total_seconds = int(duration.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    stats.append([
        "Reporting Duration (min)",
        f"{minutes}:{seconds:02d}",  # Format as MM:SS
        "primary"
    ])

    return stats


def get_overview_statistics(gpu_data):
    """
    Get the overview statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        id: string for the html href
        title: string for the title of the statistic
        list: list of plots/tables/comments that need to be displayed
    """

    # prepare data for overview stats
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

    # return the overview statistics
    return [fig_to_html(power_vs_tenso)]


def get_temp_statistics(gpu_data):
    """
    Get the temperature statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        id: string for the html href
        title: string for the title of the statistic
        list: list of plots/tables/comments that need to be displayed
    """

    # prepare data for temp stats
    gpu_data['time'] = gpu_data['timestamp'].dt.round('s')
    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]

    # prepare the plots
    plots = []
    plots.append(plot_time_series(
        time_data,
        'tmptr',
        'Mean Temperature Over Time',
        'Temperature (°C)',
        include_std=True,
    ))

    # Identify GPUs with abnormal temperatures
    gpu_temp = gpu_data['tmptr']
    temp_outliers = gpu_data[gpu_temp > gpu_temp.mean() + 3 * gpu_temp.std()]
    temp_outliers_count = temp_outliers.groupby(['node_id', 'gpu_id']).size().reset_index(name='count')

    if len(temp_outliers) > 0:
        temp_outliers_count = temp_outliers_count.sort_values('count', ascending=False)
        top_hot_gpus = temp_outliers_count.head(20)
        temp_outliers_bar = px.bar(
            top_hot_gpus,
            x='count',
            y=[f'{row['node_id']}-{row['gpu_id'][-1]}' for _, row in top_hot_gpus.iterrows()],
            labels={'count': 'Count of High Temperature Readings', 'y': 'GPU'},
            title='GPUs with Frequent High Temperatures'
        )
        temp_outliers_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        temperature_outliers_bar = fig_to_html(temp_outliers_bar)
        plots.append(temperature_outliers_bar)

        # Plot temperature for GPUs with most outliers
        temp_outliers_fig = go.Figure()
        for _, row in temp_outliers_count.head(5).iterrows():
            gpu_id = row['gpu_id']
            node = row['node_id']
            gpu_temp_data = gpu_data[(gpu_data['gpu_id'] == gpu_id) & (gpu_data['node_id'] == node)]
            temp_outliers_fig.add_trace(
                go.Scatter(
                    x=gpu_temp_data['time'],
                    y=gpu_temp_data['tmptr'],
                    opacity=0.7,
                    name=f'{node}-{gpu_id[-1]}'
                )
            )

        temp_outliers_fig.update_layout(
            title='GPUs with Most Temperature Outliers',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
        )

        temperature_outliers_plot = fig_to_html(temp_outliers_fig)
        plots.append(temperature_outliers_plot)

        #temp_outliers_count['anomaly'] = 'temperature'

    return plots


def get_power_statistics(gpu_data):
    """
    Get the power statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        id: string for the html href
        title: string for the title of the statistic
        list: list of plots/tables/comments that need to be displayed
    """

    # prepare data for power stats
    #gpu_data['time'] = gpu_data['timestamp'].dt.round('s')
    #anomalies = pd.DataFrame()
    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]

    # Power usage plots
    plots = []
    plots.append(plot_time_series(
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
    plots.append(fig_to_html(power_distribution))

    # Identify GPUs with high power usage
    gpu_power = gpu_data['power']
    power_outliers = gpu_data[gpu_power > gpu_power.mean() + 3 * gpu_power.std()]
    power_outliers_count = power_outliers.groupby(['node_id', 'gpu_id']).size().reset_index(name='count')

    if len(power_outliers_count) > 0:
        power_outliers_count = power_outliers_count.sort_values('count', ascending=False)

        # Plot Power for GPUs with most outliers
        power_outliers_fig = go.Figure()
        for _, row in power_outliers_count.head(5).iterrows():
            gpu_id = row['gpu_id']
            node = row['node_id']
            gpu_power_data = gpu_data[(gpu_data['gpu_id'] == gpu_id) & (gpu_data['node_id'] == node)]
            power_outliers_fig.add_trace(
                go.Scatter(
                    x=gpu_power_data['time'],
                    y=gpu_power_data['power'],
                    opacity=0.7,
                    name=f'{node}-{gpu_id[-1]}'
                )
            )

        power_outliers_fig.update_layout(
            title='GPUs with Most Power Outliers',
            xaxis_title='Time',
            yaxis_title='Power (W)',
        )

        plots.append(fig_to_html(power_outliers_fig))

        #power_outliers_count['anomaly'] = 'power'
        #anomalies = pd.concat([anomalies, power_outliers_count])

        return plots


def get_activity_statistics(gpu_data):
    """
    Get the activity statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        id: string for the html href
        title: string for the title of the statistic
        list: list of plots/tables/comments that need to be displayed
    """

    # prepare data for activity stats
    #gpu_data['time'] = gpu_data['timestamp'].dt.round('s')

    node_avg = gpu_data.groupby(['node_id', 'gpu_id']).agg({
        'gract': 'mean',
        'smact': 'mean',
        'tenso': 'mean',
        'tmptr': 'mean',
        'power': 'mean'
    }).reset_index()

    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]

    # prepare the plots
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
    activity_timeline.update_layout(title='GPU Activity Over Time')
    activity_timeline.update_xaxes(title_text='Time')
    activity_timeline.update_yaxes(title_text='Tensor Core Value', secondary_y=False)
    activity_timeline.update_yaxes(title_text='Graphic Activity', secondary_y=True)
    plots.append(fig_to_html(activity_timeline))

    return plots


def get_anomalies_statistics(gpu_data):
    """
    Get the anomaly statistics from the GPU data.

    Args:
        gpu_data (dict): The GPU data, raw data.

    Returns:
        id: string for the html href
        title: string for the title of the statistic
        list: list of plots/tables/comments that need to be displayed
    """

    # prepare data for activity stats
    #gpu_data['time'] = gpu_data['timestamp'].dt.round('s')

    # Power efficiency (tensor operations per watt)
    gpu_data['efficiency'] = gpu_data['tenso'] / (gpu_data['power'] + 1e-5)

    # Usage balance (ratio of SM active to tensor operations)
    gpu_data['usage_balance'] = gpu_data['smact'] / (gpu_data['tenso'] + 1e-5)

    anomalies = pd.DataFrame()

    node_avg = gpu_data.groupby(['node_id', 'gpu_id']).agg({
        'gract': 'mean',
        'smact': 'mean',
        'tenso': 'mean',
        'tmptr': 'mean',
        'power': 'mean'
    }).reset_index()

    time_data = gpu_data.groupby('time')[['gract', 'smact', 'tenso', 'tmptr', 'power']]

    # Identify GPUs with abnormal temperatures
    gpu_temp = gpu_data['tmptr']
    temp_outliers = gpu_data[gpu_temp > gpu_temp.mean() + 3 * gpu_temp.std()]
    temp_outliers_count = temp_outliers.groupby(['node_id', 'gpu_id']).size().reset_index(name='count')
    temp_outliers_count['anomaly'] = 'temperature'

    # Identify GPUs with high power usage
    gpu_power = gpu_data['power']
    power_outliers = gpu_data[gpu_power > gpu_power.mean() + 3 * gpu_power.std()]
    power_outliers_count = power_outliers.groupby(['node_id', 'gpu_id']).size().reset_index(name='count')
    power_outliers_count['anomaly'] = 'power'

    # combine the anomalies in a unique dataframe
    anomalies = pd.concat([anomalies, temp_outliers_count, power_outliers_count])
    # merge with the node_avg dataframe
    anomalies = pd.merge(anomalies, node_avg, on=['node_id', 'gpu_id'], how='left')
    # group by node and gpu, concat the 'anomaly' fileds
    anomalies = anomalies.groupby(['node_id', 'gpu_id']).agg({
        'count': 'sum',
        'anomaly': lambda x: ', '.join(x)
    }).reset_index()

    # generate table
    anomalies_table = generate_plotly_table(anomalies)

    # return the anomaly statistics
    return [anomalies_table]
