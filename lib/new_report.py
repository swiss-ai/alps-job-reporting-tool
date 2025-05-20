import argparse
import os
import re
from datetime import datetime

import jinja2
import plotly.io as pio
from plotly.basedatatypes import BaseFigure

from analysis_utils import *

pio.templates.default = 'plotly_white'


def create_report(template_file: str, output_file: str, gpu_data: pd.DataFrame) -> None:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = gpu_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    end_time = gpu_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    anomalies = find_anomalies(gpu_data)

    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    template_env.filters['slug'] = slugify
    template_env.filters['plot'] = fig_to_html

    template = template_env.get_template(template_file)

    html_content = template.render(**{
        'current_date': now,
        'start_time': start_time,
        'end_time': end_time,
        'key_statistics': get_key_statistics(gpu_data, anomalies['count'].sum()),
        'tabs': {
            'Overview': get_overview_statistics(gpu_data),
            'Temperature Analysis': get_temp_statistics(gpu_data),
            'Power Analysis': get_power_statistics(gpu_data),
            'GPUs Activity': get_activity_statistics(gpu_data),
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

    gpu_data = parse_gpu_data(input_file)
    other_data = parse_other_data(input_file2)

    print('Generating HTML report...')
    create_report('new_template.html', output_file, gpu_data)


if __name__ == '__main__':
    main()
