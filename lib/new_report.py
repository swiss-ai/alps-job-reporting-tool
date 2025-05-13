import os
import argparse
from datetime import datetime

import jinja2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# From our library
from analysis_utils import *

pio.templates.default = 'plotly_white'

# Function to analyze GPU data and generate visualizations
def analyze_gpu_data(gpu_data):
    gpu_data['time'] = gpu_data['timestamp'].dt.round('s')

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

    start_time = gpu_data['timestamp'].min()
    end_time = gpu_data['timestamp'].max()
    total_seconds = (end_time - start_time).total_seconds()

    results = {
        'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': f'{total_seconds / 60:.0f}:{total_seconds % 60:.0f}',
        'total_nodes': gpu_data['node_id'].nunique(),
        'total_gpus': len(gpu_data['gpu_id'].unique()) * gpu_data['node_id'].nunique(),
    }

    # OVERVIEW
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
    results['power_vs_tenso'] = fig_to_html(power_vs_tenso)

    # TEMPERATURE
    results['temperature_timeline'] = plot_time_series(
        time_data,
        'tmptr',
        'Mean Temperature Over Time',
        'Temperature (°C)',
        include_std=True,
    )

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
        results['temperature_outliers_bar'] = fig_to_html(temp_outliers_bar)

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

        results['temperature_outliers_plot'] = fig_to_html(temp_outliers_fig)

        temp_outliers_count['anomaly'] = 'temperature'

        anomalies = pd.concat([anomalies, temp_outliers_count])

    # POWER USAGE
    results['power_timeline'] = plot_time_series(
        time_data,
        'power',
        'Mean Power Over Time',
        'Power (W)',
        include_std=True,
    )

    # Power distribution
    power_distribution = px.histogram(
        gpu_data.groupby(['node_id', 'gpu_id']).agg({'power': 'mean'}),
        x='power',
        labels={'power': 'Power (W)', 'count': 'Count'},
        title='Mean GPU Power Usage',
    )
    results['power_distribution'] = fig_to_html(power_distribution)

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

        results['power_outliers_plot'] = fig_to_html(power_outliers_fig)

        power_outliers_count['anomaly'] = 'power'

        anomalies = pd.concat([anomalies, power_outliers_count])

    # Node efficiency analysis
    node_efficiency = gpu_data.groupby('node_id')['efficiency'].agg(['mean', 'std']).sort_values('mean')
    least_efficient_nodes = node_efficiency.head(20)

    efficiency_fig = px.bar(
        least_efficient_nodes.reset_index(),
        x='node_id',
        y='mean',
        error_y='std',
        labels={'mean': 'Mean Efficiency (TENSO/POWER)', 'node_id': 'Node ID'},
        title='Least Efficient Nodes'
    )
    results['least_efficient_nodes'] = fig_to_html(efficiency_fig)

    # ACTIVITY
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
    results['activity_timeline'] = fig_to_html(activity_timeline)

    # Add back columns to anomalies with mean of power and temp
    results['anomalies'] = anomalies.merge(node_avg, on=['node_id', 'gpu_id'], how='left')

    return results

def fig_to_html(fig):
    return fig.to_html(include_plotlyjs=False, full_html=False)

# html report generation functions
def generate_header(gpu_data):
    """
    Generate the header for the HTML report.
    """

    start_time = gpu_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_time = gpu_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    header = f"""
    <header>
        <h1>Alps Job Report</h1>
        <p>Job from {start_time} to {end_time}.</p>
        <p>Report generated on {current_date}.</p>
    </header>
    """

    return header

def generate_stat_card(title, value, type="stat-card"):
    """
    Generate a single statistic card for the HTML report.
    """

    html_card = f'''
    <div class="{type}">
        <h3>{title}</h3>
        <p>{value}</p>
    </div>
    '''

    return html_card

def generate_key_statistics(gpu_data):
    """
    Generate key statistics section of the HTML report.

    Calls the external key_statistics function which returns a list of statistics we want to display.
    """

    key_stats = get_key_statistics(gpu_data)

    html_stats = '<div class="stats-container">\n'
    for title, value, css_class in key_stats:
        html_stats += generate_stat_card(title, value, css_class)
    html_stats += '</div>'
    
    return html_stats

def generate_tabs(gpu_data):
    """
    Generate the tabs for the HTML report.
    Every tab can contain multiple plots/tables/text containers!
    """

    tabs_content = []
    # add overview data
    tabs_content.append(get_overview_statistics(gpu_data))
    # add temperature data
    tabs_content.append(get_temp_statistics(gpu_data))
    # add power data
    tabs_content.append(get_power_statistics(gpu_data))
    # add activity data
    tabs_content.append(get_activity_statistics(gpu_data))
    # add the anomalies tab
    tabs_content.append(get_anomalies_statistics(gpu_data))

    # prepare buttons structure
    html_tab_buttons = f'''
    <div class="tab-buttons">
    '''
    # prepare content structure
    html_tab_content = ""

    # iterate over the tabs we want to create
    for i, tab in enumerate(tabs_content):
        # get the data in separate variables
        tab_id, tab_title, tab_list = tab

        # add the button for the tab
        html_tab_buttons += f'''
        <a href="#{tab_id}" class="{ 'tab-button active' if i == 0 else 'tab-button' }">{tab_title}</a>
        '''

        # add the content for this tab
        html_tab_content += f'''
        <div id="{tab_id}" class="tab-content">
            <h2>{tab_title}</h2>
        '''
        
        for item in tab_list:
            html_tab_content += f'''
            <div class="plot-parent">
                {item}
            </div>
            '''

        html_tab_content += '</div>'

    # finish the html structure
    tabs = f'''
    <div class="tab-container">
    {html_tab_buttons}
    </div>
    {html_tab_content}
    </div>
    '''

    return tabs

def generate_html_report(results, template_file, output_file, gpu_data):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    template_loader = jinja2.FileSystemLoader(searchpath='./')
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template(template_file)

    html_content = template.render(**{
        **results, 
        'current_date': now, 
        'header': generate_header(gpu_data),
        'key_statistics': generate_key_statistics(gpu_data),
        'tabs': generate_tabs(gpu_data),
    })

    output_path = output_file

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f'Report generated: {output_path}')


def main():
    # Parse command-line arguments
    #parser = argparse.ArgumentParser(description="Analyze GPU data and generate an HTML report.")
    #parser.add_argument('--input_file', type=str, help="Path to the input Parquet file.")
    #parser.add_argument('--template_file', type=str, help="Path to the Jinja2 template file.")
    #parser.add_argument('--output_file', type=str, default='gpu_report.html', help="Path to the output HTML report file.")
    #args = parser.parse_args()

    #input_file = args.input_file
    #template_file = args.template_file
    #output_file = args.output_file

    input_file = "../outputs/434267_12-05-2025/data_434267_12-05-2025.parquet"
    template_file = "./new_template.html"
    output_file = "../outputs/434267_12-05-2025/434267_new_report.html"

    print('Parsing GPU data...')
    gpu_data = parse_gpu_data(input_file)

    print('Analyzing GPU data...')
    results = analyze_gpu_data(gpu_data)

    print('Generating HTML report...')
    generate_html_report(results, template_file, output_file, gpu_data)

    print('Analysis complete!')
    print(f'Report has been generated at: {output_file}')

if __name__ == '__main__':
    main()
