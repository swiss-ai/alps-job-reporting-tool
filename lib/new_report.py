import argparse
import os
import re
from datetime import datetime

import jinja2
import plotly.io as pio
from plotly.basedatatypes import BaseFigure

from analysis_utils import *

pio.templates.default = 'plotly_white'


def create_report(template_file: str, output_file: str, input_file:str, input_file2:str) -> None:
    gpu_data, pivot_gpu_data = parse_gpu_data(input_file)
    other_data = parse_other_data(input_file2)

    df = pd.read_parquet(input_file)
    df.reset_index(inplace=True)
    df.drop_duplicates(['timestamp','node_id','gpu_id'], inplace=True)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = gpu_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_time = gpu_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    anomalies = find_anomalies(gpu_data)

    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    template_env.filters['slug'] = slugify
    template_env.filters['plot'] = fig_to_html

    template = template_env.get_template(template_file)

    cpu_stats = get_cpu_statistics(other_data)

    html_content = template.render(**{
        'current_date': now,
        'start_time': start_time,
        'end_time': end_time,
        'key_statistics': get_key_statistics(gpu_data, anomalies['count'].sum()),
        'tabs': {
            'Overview': get_overview_statistics(gpu_data,df),
            'Temperature Analysis': get_temp_statistics(pivot_gpu_data),
            'Power Analysis': get_power_statistics(pivot_gpu_data),
            'GPUs Activity': get_activity_statistics(gpu_data),
            'Utilisation':get_utilisation_statistics(pivot_gpu_data),
            'NVLink Errors': get_nvlink_statistics(pivot_gpu_data),
            'CPU Current Analysis': cpu_stats[0],
            'CPU Power Analysis': cpu_stats[1],
            'CPU Temperature Analysis': cpu_stats[2],
            'Net Activity': get_net_statistics(other_data),
            'Anomalies': [anomalies, 'Hello World!'],
        },
    })

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f'Report generated: {os.path.abspath(output_file)}')


def slugify(value: str) -> str:
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def fig_to_html(figure: BaseFigure) -> str:
    return figure.to_html(include_plotlyjs=False, full_html=False)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze GPU data and generate an HTML report.')
    parser.add_argument('--input_file', type=str, help='Path to the input Parquet file containing the GPU data.')
    parser.add_argument('--input_file2', type=str, help='Path to the input Parquet file containing all other data.')
    parser.add_argument('--output_file', type=str, default='gpu_report', help='Path to the output HTML report file.')
    args = parser.parse_args()

    input_file = args.input_file
    input_file2 = args.input_file2

    # Add the timestamp to the output file name
    output_file = args.output_file
    if not output_file.endswith('.html'):
        output_file += f'_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html'

    print('Generating HTML report...')
    create_report('new_template.html', output_file, input_file, input_file2)


if __name__ == '__main__':
    main()
