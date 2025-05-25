import argparse
import json
import os
import re
from datetime import datetime

import jinja2
import plotly.io as pio
from plotly.basedatatypes import BaseFigure

from analysis_utils import *

pio.templates.default = 'plotly_white'


def create_report(template_file: str, output_file: str, input_file: str, input_file2: str) -> None:
    gpu_data, pivot_gpu_data = parse_gpu_data(input_file)
    other_data = parse_other_data(input_file2)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = gpu_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_time = gpu_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')

    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    template_env.filters['slug'] = slugify
    template_env.filters['plot'] = fig_to_html
    template_env.filters['dataframe'] = dataframe_to_table

    template = template_env.get_template(template_file)

    html_content = template.render(**{
        'current_date': now,
        'start_time': start_time,
        'end_time': end_time,
        'key_statistics': get_key_statistics(gpu_data, other_data),
        'categories': {
            'GPU Metrics': {
                'Overview': gpu_overview_statistics(gpu_data),
                'Temperature': gpu_temp_statistics(pivot_gpu_data),
                'Power': gpu_power_statistics(pivot_gpu_data),
                'Utilization': gpu_utilization_statistics(pivot_gpu_data),
                'NVLink': gpu_nvlink_statistics(pivot_gpu_data),
            },
            'CPU Metrics': {
                'Nodes': cpu_overview_statistics(other_data),
                'Current Usage': cpu_statistics(other_data, 'curr', 'Current', 'A'),
                'Power Consumption': cpu_statistics(other_data, 'power', 'Power', 'A'),
                'Temperature Evolution': cpu_statistics(other_data, 'temp', 'Temperature', '°C'),
            },
            'Network & I/O': {
                'Network Activity': net_statistics(other_data),
                'I/O Activity': io_statistics(other_data),
            },
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


def dataframe_to_table(df: pd.DataFrame) -> str:
    col_widths = {'Node ID': 100, 'GPU ID': 85}
    col = map(lambda x: dict(name=x, width=f'{col_widths.get(x, 160)}px'), df.columns)
    rows = df.dropna().round(3).values.tolist()
    data = dict(columns=list(col), data=rows, sort=True)
    # Output minified JSON, used for Grid.js rendering
    return json.dumps(data, separators=(',', ':'))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze metrics and generate an HTML report.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the Parquet file containing GPU data.')
    parser.add_argument('--input_file2', type=str, required=True,
                        help='Path to the Parquet file containing all other data.')
    parser.add_argument('--output_file', type=str, required=True, default='gpu_report',
                        help='Path to the output HTML report file.')
    args = parser.parse_args()

    input_file = args.input_file
    input_file2 = args.input_file2

    # Add the timestamp to the output file name if it doesn't end with .html
    output_file = args.output_file
    if not output_file.endswith('.html'):
        output_file += f'_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html'

    print('Generating HTML report...')
    create_report('report_template.html', output_file, input_file, input_file2)


if __name__ == '__main__':
    main()
