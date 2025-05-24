"""
OpenHands Metrics Visualizer

This module provides visualization tools for OpenHands performance metrics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional


class MetricsVisualizer:
    """
    Visualizes OpenHands performance metrics from a report.
    """
    
    def __init__(self, report_path: str = None, report_data: Dict = None):
        """
        Initialize the visualizer with either a path to a report JSON file or direct report data.
        
        Args:
            report_path: Path to the JSON file containing the metrics report
            report_data: Direct dictionary of report data
        """
        if report_path:
            with open(report_path, 'r', encoding='utf-8') as f:
                self.report = json.load(f)
        elif report_data:
            self.report = report_data
        else:
            raise ValueError("Either report_path or report_data must be provided")
        
        # Set default style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def create_dashboard(self, output_path: str):
        """
        Create a comprehensive dashboard with all key metrics.
        
        Args:
            output_path: Path to save the dashboard image
        """
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Task Completion Rate (Pie Chart)
        ax1 = plt.subplot2grid((3, 3), (0, 0))
        self._plot_completion_rate(ax1)
        
        # 2. Solution Quality (Bar Chart)
        ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
        self._plot_solution_quality(ax2)
        
        # 3. Time Efficiency (Bar Chart)
        ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        self._plot_time_efficiency(ax3)
        
        # 4. Error Recovery (Bar Chart)
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        self._plot_error_recovery(ax4)
        
        # 5. Tool Usage (Horizontal Bar Chart)
        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self._plot_tool_usage(ax5)
        
        # Add title and adjust layout
        plt.suptitle('OpenHands Performance Metrics Dashboard', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to {output_path}")
    
    def _plot_completion_rate(self, ax):
        """Plot task completion rate as a pie chart."""
        completion_rate = self.report['task_summary']['completion_rate']
        labels = ['Completed', 'Failed']
        sizes = [completion_rate, 100 - completion_rate]
        colors = ['#66b3ff', '#ff9999']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title('Task Completion Rate')
    
    def _plot_solution_quality(self, ax):
        """Plot solution quality metrics as a bar chart."""
        metrics = [
            self.report['solution_quality']['accuracy'],
            self.report['solution_quality']['robustness'],
            100 - self.report['solution_quality']['error_rate']
        ]
        labels = ['Accuracy', 'Robustness', 'Success Rate']
        
        bars = ax.bar(labels, metrics, color='#66b3ff')
        ax.set_ylim(0, 100)
        ax.set_title('Solution Quality Metrics')
        ax.set_ylabel('Percentage')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    def _plot_time_efficiency(self, ax):
        """Plot time efficiency metrics as a bar chart."""
        metrics = [
            self.report['time_efficiency']['avg_duration'],
            self.report['time_efficiency']['median_duration'],
            self.report['time_efficiency']['min_duration'],
            self.report['time_efficiency']['max_duration']
        ]
        labels = ['Average', 'Median', 'Minimum', 'Maximum']
        
        bars = ax.bar(labels, metrics, color='#66b3ff')
        ax.set_title('Task Duration (seconds)')
        ax.set_ylabel('Seconds')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}s', ha='center', va='bottom')
    
    def _plot_error_recovery(self, ax):
        """Plot error recovery metrics as a bar chart."""
        recovery_rate = self.report['error_recovery']['recovery_rate']
        
        ax.bar(['Recovery Rate'], [recovery_rate], color='#66b3ff')
        ax.set_ylim(0, 100)
        ax.set_title('Error Recovery Rate')
        ax.set_ylabel('Percentage')
        
        # Add value label
        ax.text(0, recovery_rate + 1, f'{recovery_rate:.1f}%', ha='center', va='bottom')
        
        # Add text with additional info
        tasks_with_errors = self.report['error_recovery']['tasks_with_errors']
        recovered_tasks = self.report['error_recovery']['recovered_tasks']
        ax.text(0, 50, f'Tasks with errors: {tasks_with_errors}\nRecovered tasks: {recovered_tasks}',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    def _plot_tool_usage(self, ax):
        """Plot tool usage distribution as a horizontal bar chart."""
        tool_usage = self.report['resource_usage']['tool_usage_distribution']
        tools = list(tool_usage.keys())
        counts = list(tool_usage.values())
        
        # Sort by count and limit to top 10
        if len(tools) > 10:
            sorted_indices = np.argsort(counts)[::-1][:10]
            tools = [tools[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
        else:
            sorted_indices = np.argsort(counts)[::-1]
            tools = [tools[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
        
        # Plot horizontal bar chart
        bars = ax.barh(tools, counts, color='#66b3ff')
        ax.set_title('Top Tool Usage')
        ax.set_xlabel('Count')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{width}', ha='left', va='center')
    
    def plot_all_metrics(self, output_dir: str):
        """
        Generate individual plots for all metrics and save them to the specified directory.
        
        Args:
            output_dir: Directory to save the plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Task Completion Rate
        self.plot_completion_rate(os.path.join(output_dir, 'task_completion_rate.png'))
        
        # 2. Solution Quality
        self.plot_solution_quality(os.path.join(output_dir, 'solution_quality.png'))
        
        # 3. Time Efficiency
        self.plot_time_efficiency(os.path.join(output_dir, 'time_efficiency.png'))
        
        # 4. Resource Usage
        self.plot_resource_usage(os.path.join(output_dir, 'resource_usage.png'))
        
        # 5. Error Recovery
        self.plot_error_recovery(os.path.join(output_dir, 'error_recovery.png'))
        
        # 6. Tool Selection
        self.plot_tool_selection(os.path.join(output_dir, 'tool_selection.png'))
        
        # 7. Planning Quality
        self.plot_planning_quality(os.path.join(output_dir, 'planning_quality.png'))
        
        # 8. Code Quality
        self.plot_code_quality(os.path.join(output_dir, 'code_quality.png'))
        
        # 9. Information Retrieval
        self.plot_information_retrieval(os.path.join(output_dir, 'information_retrieval.png'))
        
        print(f"All plots saved to {output_dir}")
    
    def plot_completion_rate(self, output_path: str = None):
        """Plot and optionally save the task completion rate."""
        plt.figure(figsize=(10, 8))
        
        completion_rate = self.report['task_summary']['completion_rate']
        labels = ['Completed', 'Failed']
        sizes = [completion_rate, 100 - completion_rate]
        colors = ['#66b3ff', '#ff9999']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Task Completion Rate', fontsize=16)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_solution_quality(self, output_path: str = None):
        """Plot and optionally save the solution quality metrics."""
        plt.figure(figsize=(12, 8))
        
        metrics = [
            self.report['solution_quality']['accuracy'],
            self.report['solution_quality']['robustness'],
            100 - self.report['solution_quality']['error_rate']
        ]
        labels = ['Accuracy', 'Robustness', 'Success Rate']
        
        bars = plt.bar(labels, metrics, color='#66b3ff')
        plt.ylim(0, 100)
        plt.title('Solution Quality Metrics', fontsize=16)
        plt.ylabel('Percentage', fontsize=14)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_time_efficiency(self, output_path: str = None):
        """Plot and optionally save the time efficiency metrics."""
        plt.figure(figsize=(12, 8))
        
        metrics = [
            self.report['time_efficiency']['avg_duration'],
            self.report['time_efficiency']['median_duration'],
            self.report['time_efficiency']['min_duration'],
            self.report['time_efficiency']['max_duration']
        ]
        labels = ['Average', 'Median', 'Minimum', 'Maximum']
        
        bars = plt.bar(labels, metrics, color='#66b3ff')
        plt.title('Task Duration (seconds)', fontsize=16)
        plt.ylabel('Seconds', fontsize=14)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_resource_usage(self, output_path: str = None):
        """Plot and optionally save the resource usage metrics."""
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Plot 1: API and Tool Calls
        metrics1 = [
            self.report['resource_usage']['api_call_count'],
            self.report['resource_usage']['total_tool_calls']
        ]
        labels1 = ['API Calls', 'Tool Calls']
        
        bars1 = ax1.bar(labels1, metrics1, color=['#ff9999', '#66b3ff'])
        ax1.set_title('API and Tool Calls', fontsize=14)
        ax1.set_ylabel('Count', fontsize=12)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}', ha='center', va='bottom')
        
        # Plot 2: Tool Usage Distribution (Top 5)
        tool_usage = self.report['resource_usage']['tool_usage_distribution']
        tools = list(tool_usage.keys())
        counts = list(tool_usage.values())
        
        # Sort by count and limit to top 5
        if len(tools) > 5:
            sorted_indices = np.argsort(counts)[::-1][:5]
            tools = [tools[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
        else:
            sorted_indices = np.argsort(counts)[::-1]
            tools = [tools[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
        
        bars2 = ax2.bar(tools, counts, color='#66b3ff')
        ax2.set_title('Top 5 Tool Usage', fontsize=14)
        ax2.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_recovery(self, output_path: str = None):
        """Plot and optionally save the error recovery metrics."""
        plt.figure(figsize=(10, 8))
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Plot 1: Recovery Rate
        recovery_rate = self.report['error_recovery']['recovery_rate']
        
        ax1.bar(['Recovery Rate'], [recovery_rate], color='#66b3ff')
        ax1.set_ylim(0, 100)
        ax1.set_title('Error Recovery Rate', fontsize=14)
        ax1.set_ylabel('Percentage', fontsize=12)
        
        # Add value label
        ax1.text(0, recovery_rate + 1, f'{recovery_rate:.1f}%', ha='center', va='bottom')
        
        # Plot 2: Error Statistics
        tasks_with_errors = self.report['error_recovery']['tasks_with_errors']
        recovered_tasks = self.report['error_recovery']['recovered_tasks']
        
        ax2.bar(['Tasks with Errors', 'Recovered Tasks'], [tasks_with_errors, recovered_tasks], 
               color=['#ff9999', '#66b3ff'])
        ax2.set_title('Error Statistics', fontsize=14)
        ax2.set_ylabel('Count', fontsize=12)
        
        # Add value labels
        ax2.text(0, tasks_with_errors + 0.5, f'{tasks_with_errors}', ha='center', va='bottom')
        ax2.text(1, recovered_tasks + 0.5, f'{recovered_tasks}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_tool_selection(self, output_path: str = None):
        """Plot and optionally save the tool selection metrics."""
        plt.figure(figsize=(12, 8))
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Plot 1: Tool Selection Metrics
        metrics = [
            self.report['tool_selection']['avg_tool_diversity'],
            self.report['tool_selection']['avg_tool_repetition']
        ]
        labels = ['Avg Tool Diversity', 'Avg Tool Repetition']
        
        bars = ax1.bar(labels, metrics, color=['#66b3ff', '#ff9999'])
        ax1.set_title('Tool Selection Metrics', fontsize=14)
        ax1.set_ylabel('Count', fontsize=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Plot 2: Most Used Tools
        most_used = self.report['tool_selection']['most_used_tools']
        tools = list(most_used.keys())
        counts = list(most_used.values())
        
        bars2 = ax2.bar(tools, counts, color='#66b3ff')
        ax2.set_title('Most Used Tools', fontsize=14)
        ax2.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_planning_quality(self, output_path: str = None):
        """Plot and optionally save the planning quality metrics."""
        plt.figure(figsize=(12, 8))
        
        metrics = [
            self.report['planning_quality']['planning_ratio'] * 100,
            self.report['planning_quality']['planning_before_action_ratio'] * 100,
            self.report['planning_quality']['tasks_with_planning'] / self.report['task_summary']['total_tasks'] * 100
        ]
        labels = ['Planning Ratio', 'Planning Before Action', 'Tasks with Planning']
        
        bars = plt.bar(labels, metrics, color='#66b3ff')
        plt.ylim(0, 100)
        plt.title('Planning Quality Metrics', fontsize=16)
        plt.ylabel('Percentage', fontsize=14)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_code_quality(self, output_path: str = None):
        """Plot and optionally save the code quality metrics."""
        plt.figure(figsize=(14, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        
        # Plot 1: Code Metrics
        metrics1 = [
            self.report['code_quality']['avg_lines_per_block'],
            self.report['code_quality']['avg_complexity']
        ]
        labels1 = ['Avg Lines per Block', 'Avg Complexity']
        
        bars1 = ax1.bar(labels1, metrics1, color='#66b3ff')
        ax1.set_title('Code Metrics', fontsize=14)
        ax1.set_ylabel('Count', fontsize=12)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Plot 2: Comment Ratio
        comment_ratio = self.report['code_quality']['comment_ratio'] * 100
        
        ax2.pie([comment_ratio, 100 - comment_ratio], 
               labels=['Comments', 'Code'], 
               colors=['#ff9999', '#66b3ff'],
               autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        ax2.set_title('Comment to Code Ratio', fontsize=14)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_information_retrieval(self, output_path: str = None):
        """Plot and optionally save the information retrieval metrics."""
        plt.figure(figsize=(12, 8))
        
        metrics = [
            self.report['information_retrieval']['retrieval_ratio'] * 100,
            self.report['information_retrieval']['retrieval_before_action_ratio'] * 100,
            self.report['information_retrieval']['tasks_with_retrieval'] / self.report['task_summary']['total_tasks'] * 100
        ]
        labels = ['Retrieval Ratio', 'Retrieval Before Action', 'Tasks with Retrieval']
        
        bars = plt.bar(labels, metrics, color='#66b3ff')
        plt.ylim(0, 100)
        plt.title('Information Retrieval Metrics', fontsize=16)
        plt.ylabel('Percentage', fontsize=14)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """
    Main function to demonstrate the usage of the MetricsVisualizer.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize OpenHands metrics')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input report JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--dashboard', '-d', action='store_true', help='Create a dashboard visualization')
    parser.add_argument('--dashboard-path', type=str, default='dashboard.png', help='Path to save the dashboard')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MetricsVisualizer(report_path=args.input)
    
    # Generate all plots
    visualizer.plot_all_metrics(args.output_dir)
    
    # Generate dashboard if requested
    if args.dashboard:
        visualizer.create_dashboard(args.dashboard_path)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()