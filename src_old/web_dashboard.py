"""
OpenHands Web Dashboard

This module provides a web dashboard for visualizing OpenHands performance metrics.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class OpenHandsDashboard:
    """
    Web dashboard for OpenHands performance metrics.
    """
    
    def __init__(self, report_path: str = None, report_data: Dict = None, host: str = '0.0.0.0', port: int = 12000):
        """
        Initialize the dashboard with either a path to a report JSON file or direct report data.
        
        Args:
            report_path: Path to the JSON file containing the metrics report
            report_data: Direct dictionary of report data
            host: Host to run the dashboard on
            port: Port to run the dashboard on
        """
        if report_path:
            with open(report_path, 'r', encoding='utf-8') as f:
                self.report = json.load(f)
        elif report_data:
            self.report = report_data
        else:
            raise ValueError("Either report_path or report_data must be provided")
        
        self.host = host
        self.port = port
        self.app = self._create_app()
    
    def _create_app(self):
        """Create the Dash application."""
        app = dash.Dash(__name__, title="OpenHands Metrics Dashboard")
        
        # Define the layout
        app.layout = html.Div([
            html.H1("OpenHands Performance Metrics Dashboard", style={'textAlign': 'center'}),
            
            # Summary Cards
            html.Div([
                self._create_summary_card("Total Tasks", self.report['task_summary']['total_tasks']),
                self._create_summary_card("Completed Tasks", self.report['task_summary']['completed_tasks']),
                self._create_summary_card("Completion Rate", f"{self.report['task_summary']['completion_rate']:.1f}%"),
                self._create_summary_card("Avg Duration", f"{self.report['time_efficiency']['avg_duration']:.1f}s"),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px 0'}),
            
            # Tabs for different metric categories
            dcc.Tabs([
                # Tab 1: Task Completion and Solution Quality
                dcc.Tab(label="Task Completion & Quality", children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=self._create_completion_rate_figure())
                        ], style={'width': '40%'}),
                        html.Div([
                            dcc.Graph(figure=self._create_solution_quality_figure())
                        ], style={'width': '60%'})
                    ], style={'display': 'flex'})
                ]),
                
                # Tab 2: Time Efficiency
                dcc.Tab(label="Time Efficiency", children=[
                    dcc.Graph(figure=self._create_time_efficiency_figure())
                ]),
                
                # Tab 3: Resource Usage
                dcc.Tab(label="Resource Usage", children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=self._create_api_tool_calls_figure())
                        ], style={'width': '50%'}),
                        html.Div([
                            dcc.Graph(figure=self._create_tool_usage_figure())
                        ], style={'width': '50%'})
                    ], style={'display': 'flex'})
                ]),
                
                # Tab 4: Error Recovery
                dcc.Tab(label="Error Recovery", children=[
                    dcc.Graph(figure=self._create_error_recovery_figure())
                ]),
                
                # Tab 5: Tool Selection
                dcc.Tab(label="Tool Selection", children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=self._create_tool_selection_metrics_figure())
                        ], style={'width': '50%'}),
                        html.Div([
                            dcc.Graph(figure=self._create_most_used_tools_figure())
                        ], style={'width': '50%'})
                    ], style={'display': 'flex'})
                ]),
                
                # Tab 6: Planning Quality
                dcc.Tab(label="Planning Quality", children=[
                    dcc.Graph(figure=self._create_planning_quality_figure())
                ]),
                
                # Tab 7: Code Quality
                dcc.Tab(label="Code Quality", children=[
                    html.Div([
                        html.Div([
                            dcc.Graph(figure=self._create_code_metrics_figure())
                        ], style={'width': '50%'}),
                        html.Div([
                            dcc.Graph(figure=self._create_comment_ratio_figure())
                        ], style={'width': '50%'})
                    ], style={'display': 'flex'})
                ]),
                
                # Tab 8: Information Retrieval
                dcc.Tab(label="Information Retrieval", children=[
                    dcc.Graph(figure=self._create_information_retrieval_figure())
                ])
            ])
        ], style={'padding': '20px'})
        
        return app
    
    def _create_summary_card(self, title, value):
        """Create a summary card with a title and value."""
        return html.Div([
            html.H4(title),
            html.H2(value)
        ], style={
            'textAlign': 'center',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '10px',
            'padding': '15px',
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',
            'width': '20%'
        })
    
    def _create_completion_rate_figure(self):
        """Create a pie chart for task completion rate."""
        completion_rate = self.report['task_summary']['completion_rate']
        
        fig = go.Figure(data=[go.Pie(
            labels=['Completed', 'Failed'],
            values=[completion_rate, 100 - completion_rate],
            hole=.3,
            marker_colors=['#66b3ff', '#ff9999']
        )])
        
        fig.update_layout(
            title_text="Task Completion Rate",
            showlegend=True
        )
        
        return fig
    
    def _create_solution_quality_figure(self):
        """Create a bar chart for solution quality metrics."""
        metrics = [
            self.report['solution_quality']['accuracy'],
            self.report['solution_quality']['robustness'],
            100 - self.report['solution_quality']['error_rate']
        ]
        labels = ['Accuracy', 'Robustness', 'Success Rate']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color='#66b3ff',
            text=[f'{m:.1f}%' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Solution Quality Metrics",
            yaxis=dict(
                title="Percentage",
                range=[0, 100]
            )
        )
        
        return fig
    
    def _create_time_efficiency_figure(self):
        """Create a bar chart for time efficiency metrics."""
        metrics = [
            self.report['time_efficiency']['avg_duration'],
            self.report['time_efficiency']['median_duration'],
            self.report['time_efficiency']['min_duration'],
            self.report['time_efficiency']['max_duration']
        ]
        labels = ['Average', 'Median', 'Minimum', 'Maximum']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color='#66b3ff',
            text=[f'{m:.1f}s' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Task Duration (seconds)",
            yaxis=dict(
                title="Seconds"
            )
        )
        
        return fig
    
    def _create_api_tool_calls_figure(self):
        """Create a bar chart for API and tool calls."""
        metrics = [
            self.report['resource_usage']['api_call_count'],
            self.report['resource_usage']['total_tool_calls']
        ]
        labels = ['API Calls', 'Tool Calls']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color=['#ff9999', '#66b3ff'],
            text=[f'{m}' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="API and Tool Calls",
            yaxis=dict(
                title="Count"
            )
        )
        
        return fig
    
    def _create_tool_usage_figure(self):
        """Create a bar chart for tool usage distribution."""
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
        
        fig = go.Figure(data=[go.Bar(
            x=tools,
            y=counts,
            marker_color='#66b3ff',
            text=[f'{c}' for c in counts],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Tool Usage Distribution",
            yaxis=dict(
                title="Count"
            ),
            xaxis=dict(
                tickangle=45
            )
        )
        
        return fig
    
    def _create_error_recovery_figure(self):
        """Create a figure for error recovery metrics."""
        # Create a figure with subplots
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'bar'}]])
        
        # Add pie chart for recovery rate
        recovery_rate = self.report['error_recovery']['recovery_rate']
        fig.add_trace(go.Pie(
            labels=['Recovered', 'Not Recovered'],
            values=[recovery_rate, 100 - recovery_rate],
            hole=.3,
            marker_colors=['#66b3ff', '#ff9999'],
            name="Recovery Rate"
        ), row=1, col=1)
        
        # Add bar chart for error statistics
        tasks_with_errors = self.report['error_recovery']['tasks_with_errors']
        recovered_tasks = self.report['error_recovery']['recovered_tasks']
        fig.add_trace(go.Bar(
            x=['Tasks with Errors', 'Recovered Tasks'],
            y=[tasks_with_errors, recovered_tasks],
            marker_color=['#ff9999', '#66b3ff'],
            text=[f'{tasks_with_errors}', f'{recovered_tasks}'],
            textposition='auto',
            name="Error Statistics"
        ), row=1, col=2)
        
        fig.update_layout(
            title_text="Error Recovery Metrics"
        )
        
        return fig
    
    def _create_tool_selection_metrics_figure(self):
        """Create a bar chart for tool selection metrics."""
        metrics = [
            self.report['tool_selection']['avg_tool_diversity'],
            self.report['tool_selection']['avg_tool_repetition']
        ]
        labels = ['Avg Tool Diversity', 'Avg Tool Repetition']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color=['#66b3ff', '#ff9999'],
            text=[f'{m:.1f}' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Tool Selection Metrics",
            yaxis=dict(
                title="Count"
            )
        )
        
        return fig
    
    def _create_most_used_tools_figure(self):
        """Create a bar chart for most used tools."""
        most_used = self.report['tool_selection']['most_used_tools']
        tools = list(most_used.keys())
        counts = list(most_used.values())
        
        fig = go.Figure(data=[go.Bar(
            x=tools,
            y=counts,
            marker_color='#66b3ff',
            text=[f'{c}' for c in counts],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Most Used Tools",
            yaxis=dict(
                title="Count"
            ),
            xaxis=dict(
                tickangle=45
            )
        )
        
        return fig
    
    def _create_planning_quality_figure(self):
        """Create a bar chart for planning quality metrics."""
        metrics = [
            self.report['planning_quality']['planning_ratio'] * 100,
            self.report['planning_quality']['planning_before_action_ratio'] * 100,
            self.report['planning_quality']['tasks_with_planning'] / self.report['task_summary']['total_tasks'] * 100
        ]
        labels = ['Planning Ratio', 'Planning Before Action', 'Tasks with Planning']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color='#66b3ff',
            text=[f'{m:.1f}%' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Planning Quality Metrics",
            yaxis=dict(
                title="Percentage",
                range=[0, 100]
            )
        )
        
        return fig
    
    def _create_code_metrics_figure(self):
        """Create a bar chart for code metrics."""
        metrics = [
            self.report['code_quality']['avg_lines_per_block'],
            self.report['code_quality']['avg_complexity']
        ]
        labels = ['Avg Lines per Block', 'Avg Complexity']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color='#66b3ff',
            text=[f'{m:.1f}' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Code Metrics",
            yaxis=dict(
                title="Count"
            )
        )
        
        return fig
    
    def _create_comment_ratio_figure(self):
        """Create a pie chart for comment ratio."""
        comment_ratio = self.report['code_quality']['comment_ratio'] * 100
        
        fig = go.Figure(data=[go.Pie(
            labels=['Comments', 'Code'],
            values=[comment_ratio, 100 - comment_ratio],
            hole=.3,
            marker_colors=['#ff9999', '#66b3ff']
        )])
        
        fig.update_layout(
            title_text="Comment to Code Ratio"
        )
        
        return fig
    
    def _create_information_retrieval_figure(self):
        """Create a bar chart for information retrieval metrics."""
        metrics = [
            self.report['information_retrieval']['retrieval_ratio'] * 100,
            self.report['information_retrieval']['retrieval_before_action_ratio'] * 100,
            self.report['information_retrieval']['tasks_with_retrieval'] / self.report['task_summary']['total_tasks'] * 100
        ]
        labels = ['Retrieval Ratio', 'Retrieval Before Action', 'Tasks with Retrieval']
        
        fig = go.Figure(data=[go.Bar(
            x=labels,
            y=metrics,
            marker_color='#66b3ff',
            text=[f'{m:.1f}%' for m in metrics],
            textposition='auto'
        )])
        
        fig.update_layout(
            title_text="Information Retrieval Metrics",
            yaxis=dict(
                title="Percentage",
                range=[0, 100]
            )
        )
        
        return fig
    
    def run(self):
        """Run the dashboard server."""
        self.app.run(host=self.host, port=self.port, debug=False)


def main():
    """
    Main function to run the OpenHands dashboard.
    """
    parser = argparse.ArgumentParser(description='Run OpenHands metrics dashboard')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input report JSON file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    parser.add_argument('--port', type=int, default=12000, help='Port to run the dashboard on')
    
    args = parser.parse_args()
    
    # Create and run dashboard
    dashboard = OpenHandsDashboard(report_path=args.input, host=args.host, port=args.port)
    print(f"Dashboard running at http://{args.host}:{args.port}/")
    dashboard.run()


if __name__ == "__main__":
    main()