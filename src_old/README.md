# OpenHands Data Processing Module

This module processes OpenHands event stream data to calculate various performance metrics related to task completion, solution quality, efficiency, and other indicators.

## Features

- **Comprehensive Metrics**: Analyzes OpenHands event stream data to calculate metrics for:
  - Task completion rate
  - Solution quality (accuracy, efficiency, robustness)
  - Time efficiency
  - Resource usage
  - Error recovery
  - Tool selection efficiency
  - Planning quality
  - Code quality
  - Information retrieval efficiency

- **Visualization**: Generates visualizations for all metrics using matplotlib and seaborn

- **Web Dashboard**: Provides an interactive web dashboard using Dash and Plotly

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd data_process
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Process an OpenHands event stream data file and generate a report:

```bash
python main.py --input data/test.json --output-dir output
```

### Generate Visualizations

Process the data and generate visualizations:

```bash
python main.py --input data/test.json --output-dir output --visualize
```

### Generate Dashboard

Process the data and generate a dashboard visualization:

```bash
python main.py --input data/test.json --output-dir output --visualize --dashboard
```

### Run Web Dashboard

Process the data and run an interactive web dashboard:

```bash
python main.py --input data/test.json --output-dir output --web-dashboard --port 12000
```

### Command-line Options

- `--input`, `-i`: Path to the input JSON file with OpenHands event stream data (required)
- `--output-dir`, `-o`: Directory to save output files (default: 'output')
- `--report-only`: Generate only the report without visualizations
- `--visualize`, `-v`: Generate visualizations
- `--dashboard`, `-d`: Create a dashboard visualization
- `--web-dashboard`, `-w`: Run the web dashboard
- `--host`: Host to run the web dashboard on (default: '0.0.0.0')
- `--port`: Port to run the web dashboard on (default: 12000)

## Metrics Calculated

### 1. Outcome Metrics

#### 1.1 Task Completion Rate
- Percentage of successfully completed tasks

#### 1.2 Solution Quality
- Accuracy: Percentage of tasks completed correctly
- Efficiency: Average number of actions per completed task
- Robustness: Ratio of errors to total actions

#### 1.3 Time Efficiency
- Average, median, minimum, and maximum task duration

#### 1.4 Resource Usage
- API call count
- Total tool calls
- Average tool calls per task
- Tool usage distribution

#### 1.5 Error Recovery
- Tasks with errors
- Recovered tasks
- Recovery rate

#### 1.6 Consistency
- Measured through solution quality metrics across different tasks

### 2. Process Metrics

#### 2.1 Step-wise Correctness
- Analyzed through error rates and recovery metrics

#### 2.2 Tool Selection Efficiency
- Average tool diversity per task
- Average tool repetition (potentially inefficient tool usage)
- Most used tools

#### 2.3 Planning Quality
- Planning ratio (planning events to total actions)
- Average planning events per task
- Planning before action ratio

#### 2.4 Information Retrieval Efficiency
- Retrieval ratio
- Average retrieval actions per task
- Retrieval before action ratio

#### 2.5 Code Quality
- Total code blocks
- Average lines per block
- Average complexity
- Comment ratio
- Total lines of code

#### 2.6 Learning Adaptability
- Inferred from performance improvements in similar tasks

## API Usage

You can also use the modules programmatically:

```python
from src.data_processor import OpenHandsDataProcessor
from src.visualizer import MetricsVisualizer

# Process data
processor = OpenHandsDataProcessor(data_path='data/test.json')
report = processor.generate_comprehensive_report()

# Save report
processor.save_report('output/report.json')

# Generate visualizations
visualizer = MetricsVisualizer(report_data=report)
visualizer.plot_all_metrics('output/visualizations')
visualizer.create_dashboard('output/dashboard.png')
```

## Web Dashboard

The web dashboard provides an interactive interface to explore the metrics. To run it:

```bash
python main.py --input data/test.json --web-dashboard --port 12000
```

Then open your browser and navigate to `http://localhost:12000` to view the dashboard.
