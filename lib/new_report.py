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

def generate_html_report(template_file, output_file, gpu_data):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    template_loader = jinja2.FileSystemLoader(searchpath='./')
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template(template_file)

    html_content = template.render(**{
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

    print('Generating HTML report...')
    generate_html_report(template_file, output_file, gpu_data)

    print('Analysis complete!')
    print(f'Report has been generated at: {output_file}')

if __name__ == '__main__':
    main()
