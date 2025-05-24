#!/usr/bin/env python3
"""
OpenHands Data Processing Pipeline

This script processes OpenHands event stream data and generates metrics reports and visualizations.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import OpenHandsDataProcessor
from visualizer import MetricsVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process OpenHands event stream data')
    
    # Input/output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the input JSON file with OpenHands event stream data')
    parser.add_argument('--output-dir', '-o', type=str, default='output',
                        help='Directory to save output files')
    
    # Processing options
    parser.add_argument('--report-only', action='store_true',
                        help='Generate only the report without visualizations')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--dashboard', '-d', action='store_true',
                        help='Create a dashboard visualization')
    
    # Web dashboard options
    parser.add_argument('--web-dashboard', '-w', action='store_true',
                        help='Run the web dashboard')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the web dashboard on')
    parser.add_argument('--port', type=int, default=12000,
                        help='Port to run the web dashboard on')
    
    return parser.parse_args()


def main():
    """Main function to run the data processing pipeline."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Process the data
    print(f"Processing data from {args.input}...")
    processor = OpenHandsDataProcessor(data_path=args.input)
    
    # Generate and save the report
    report_path = os.path.join(args.output_dir, f'report_{timestamp}.json')
    processor.save_report(report_path)
    print(f"Report saved to {report_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        print("Generating visualizations...")
        visualizer = MetricsVisualizer(report_path=report_path)
        visualizer.plot_all_metrics(viz_dir)
        
        if args.dashboard:
            dashboard_path = os.path.join(args.output_dir, f'dashboard_{timestamp}.png')
            visualizer.create_dashboard(dashboard_path)
            print(f"Dashboard saved to {dashboard_path}")
    
    # Run web dashboard if requested
    if args.web_dashboard:
        try:
            from web_dashboard import OpenHandsDashboard
            
            print(f"Starting web dashboard at http://{args.host}:{args.port}/")
            dashboard = OpenHandsDashboard(report_path=report_path, host=args.host, port=args.port)
            dashboard.run()
        except ImportError as e:
            print(f"Error: {e}")
            print("Web dashboard requires additional dependencies. Install them with:")
            print("pip install dash plotly")
    
    # Print summary
    report = processor.generate_comprehensive_report()
    print("\nSummary:")
    print(f"Total Tasks: {report['task_summary']['total_tasks']}")
    print(f"Completed Tasks: {report['task_summary']['completed_tasks']}")
    print(f"Completion Rate: {report['task_summary']['completion_rate']:.2f}%")
    print(f"Average Task Duration: {report['time_efficiency']['avg_duration']:.2f} seconds")
    print(f"Most Used Tools: {', '.join(list(report['tool_selection']['most_used_tools'].keys())[:3])}")


if __name__ == "__main__":
    main()