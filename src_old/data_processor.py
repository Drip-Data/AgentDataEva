"""
OpenHands Data Processor

This module processes OpenHands event stream data to calculate various metrics
related to task completion, solution quality, efficiency, and other performance indicators.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


class OpenHandsDataProcessor:
    """
    Processes OpenHands event stream data to calculate performance metrics.
    """
    
    def __init__(self, data_path: str = None, data: List[Dict] = None):
        """
        Initialize the data processor with either a path to a JSON file or direct data.
        
        Args:
            data_path: Path to the JSON file containing OpenHands event stream data
            data: Direct list of event dictionaries
        """
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        elif data:
            self.data = data
        else:
            raise ValueError("Either data_path or data must be provided")
        
        # Initialize processed data containers
        self.tasks = []
        self.actions = []
        self.events_by_task = defaultdict(list)
        self.task_completion_status = {}
        self.tool_usage = defaultdict(int)
        self.error_events = []
        self.task_durations = {}
        self.api_calls = defaultdict(int)
        
        # Process the data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        Preprocess the raw data to extract tasks, actions, and other relevant information.
        """
        current_task_id = None
        task_start_time = None
        
        for event in self.data:
            # Track task boundaries
            if event.get('source') == 'user' and event.get('action') == 'message':
                # New task starts with user message
                current_task_id = event.get('id')
                task_start_time = datetime.fromisoformat(event.get('timestamp'))
                self.tasks.append({
                    'id': current_task_id,
                    'start_time': task_start_time,
                    'message': event.get('message', '')
                })
            
            # Track task completion
            if event.get('action') == 'finish':
                if 'args' in event and 'task_completed' in event['args']:
                    task_completed = event['args']['task_completed'] == 'true'
                    self.task_completion_status[current_task_id] = task_completed
                    
                    # Calculate task duration
                    end_time = datetime.fromisoformat(event.get('timestamp'))
                    if task_start_time and current_task_id:
                        duration = (end_time - task_start_time).total_seconds()
                        self.task_durations[current_task_id] = duration
            
            # Track actions
            if event.get('source') == 'agent' and 'action' in event:
                action_type = event.get('action')
                self.actions.append({
                    'id': event.get('id'),
                    'task_id': current_task_id,
                    'action_type': action_type,
                    'timestamp': event.get('timestamp')
                })
                
                # Track tool usage
                if action_type == 'run' and 'args' in event and 'name' in event['args']:
                    tool_name = event['args']['name']
                    self.tool_usage[tool_name] += 1
                    
                    # Track API calls
                    if tool_name in ['web_read', 'browser']:
                        self.api_calls['external_api'] += 1
            
            # Track errors
            if 'error' in event or (event.get('observation') and 'error' in str(event.get('observation')).lower()):
                self.error_events.append(event)
            
            # Group events by task
            if current_task_id:
                self.events_by_task[current_task_id].append(event)
    
    def calculate_task_completion_rate(self) -> float:
        """
        Calculate the percentage of successfully completed tasks.
        
        Returns:
            float: Task completion rate as a percentage
        """
        if not self.task_completion_status:
            return 0.0
        
        completed_tasks = sum(1 for status in self.task_completion_status.values() if status)
        return (completed_tasks / len(self.task_completion_status)) * 100
    
    def calculate_solution_quality(self) -> Dict[str, Any]:
        """
        Evaluate the quality of solutions based on accuracy, efficiency, and robustness.
        
        Returns:
            Dict: Metrics related to solution quality
        """
        # Accuracy: Percentage of tasks completed correctly
        accuracy = self.calculate_task_completion_rate()
        
        # Efficiency: Average number of actions per completed task
        actions_per_task = defaultdict(int)
        for action in self.actions:
            task_id = action.get('task_id')
            if task_id:
                actions_per_task[task_id] += 1
        
        completed_task_ids = [tid for tid, completed in self.task_completion_status.items() if completed]
        avg_actions = 0
        if completed_task_ids:
            avg_actions = sum(actions_per_task.get(tid, 0) for tid in completed_task_ids) / len(completed_task_ids)
        
        # Robustness: Ratio of errors to total actions
        error_rate = len(self.error_events) / max(1, len(self.actions))
        robustness = 1 - error_rate  # Higher is better
        
        return {
            'accuracy': accuracy,
            'efficiency': avg_actions,
            'robustness': robustness * 100,  # as percentage
            'error_rate': error_rate * 100   # as percentage
        }
    
    def calculate_time_efficiency(self) -> Dict[str, Any]:
        """
        Calculate metrics related to time efficiency.
        
        Returns:
            Dict: Time efficiency metrics
        """
        if not self.task_durations:
            return {
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'median_duration': 0
            }
        
        durations = list(self.task_durations.values())
        
        return {
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'median_duration': sorted(durations)[len(durations) // 2]
        }
    
    def calculate_resource_usage(self) -> Dict[str, Any]:
        """
        Calculate metrics related to resource usage.
        
        Returns:
            Dict: Resource usage metrics
        """
        # Count API calls
        api_call_count = sum(self.api_calls.values())
        
        # Count tool usage
        total_tool_calls = sum(self.tool_usage.values())
        
        # Calculate average tool calls per task
        avg_tool_calls = total_tool_calls / max(1, len(self.tasks))
        
        return {
            'api_call_count': api_call_count,
            'total_tool_calls': total_tool_calls,
            'avg_tool_calls_per_task': avg_tool_calls,
            'tool_usage_distribution': dict(self.tool_usage)
        }
    
    def calculate_error_recovery(self) -> Dict[str, Any]:
        """
        Calculate metrics related to error recovery.
        
        Returns:
            Dict: Error recovery metrics
        """
        # Count tasks with errors
        tasks_with_errors = set()
        for event in self.error_events:
            for task_id, events in self.events_by_task.items():
                if event in events:
                    tasks_with_errors.add(task_id)
        
        # Count tasks with errors that were still completed
        recovered_tasks = sum(1 for tid in tasks_with_errors if self.task_completion_status.get(tid, False))
        
        # Calculate recovery rate
        recovery_rate = 0
        if tasks_with_errors:
            recovery_rate = (recovered_tasks / len(tasks_with_errors)) * 100
        
        return {
            'tasks_with_errors': len(tasks_with_errors),
            'recovered_tasks': recovered_tasks,
            'recovery_rate': recovery_rate
        }
    
    def analyze_tool_selection(self) -> Dict[str, Any]:
        """
        Analyze the efficiency of tool selection.
        
        Returns:
            Dict: Tool selection efficiency metrics
        """
        # Count tool usage by task
        tool_usage_by_task = defaultdict(lambda: defaultdict(int))
        
        for event in self.data:
            if event.get('source') == 'agent' and event.get('action') == 'run':
                task_id = None
                # Find the task this event belongs to
                for tid, events in self.events_by_task.items():
                    if event in events:
                        task_id = tid
                        break
                
                if task_id and 'args' in event and 'name' in event['args']:
                    tool_name = event['args']['name']
                    tool_usage_by_task[task_id][tool_name] += 1
        
        # Calculate tool diversity per task
        tool_diversity = {tid: len(tools) for tid, tools in tool_usage_by_task.items()}
        avg_tool_diversity = sum(tool_diversity.values()) / max(1, len(tool_diversity))
        
        # Calculate tool repetition (potentially inefficient tool usage)
        tool_repetition = {tid: sum(count > 2 for tool, count in tools.items()) 
                          for tid, tools in tool_usage_by_task.items()}
        avg_tool_repetition = sum(tool_repetition.values()) / max(1, len(tool_repetition))
        
        return {
            'avg_tool_diversity': avg_tool_diversity,
            'avg_tool_repetition': avg_tool_repetition,
            'tool_diversity_by_task': tool_diversity,
            'most_used_tools': dict(sorted(self.tool_usage.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def analyze_planning_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality of planning in the solution approach.
        
        Returns:
            Dict: Planning quality metrics
        """
        # Look for planning-related events (e.g., 'think' tool usage)
        planning_events = []
        for event in self.data:
            if (event.get('source') == 'agent' and event.get('action') == 'run' and 
                'args' in event and event['args'].get('name') == 'think'):
                planning_events.append(event)
        
        # Calculate planning ratio (planning events to total actions)
        planning_ratio = len(planning_events) / max(1, len(self.actions))
        
        # Calculate average planning events per task
        planning_by_task = defaultdict(int)
        for event in planning_events:
            for task_id, events in self.events_by_task.items():
                if event in events:
                    planning_by_task[task_id] += 1
        
        avg_planning_per_task = sum(planning_by_task.values()) / max(1, len(self.tasks))
        
        # Calculate planning before action ratio
        # (This is a simplified metric - in a real system, you'd want to analyze the sequence more carefully)
        planning_before_action = 0
        for task_id, events in self.events_by_task.items():
            action_events = [e for e in events if e.get('source') == 'agent' and e.get('action') != 'run']
            if action_events:
                first_action_idx = events.index(action_events[0])
                planning_before_first_action = sum(1 for e in events[:first_action_idx] 
                                                if e.get('source') == 'agent' and 
                                                e.get('action') == 'run' and 
                                                'args' in e and 
                                                e['args'].get('name') == 'think')
                if planning_before_first_action > 0:
                    planning_before_action += 1
        
        planning_before_action_ratio = planning_before_action / max(1, len(self.tasks))
        
        return {
            'planning_ratio': planning_ratio,
            'avg_planning_per_task': avg_planning_per_task,
            'planning_before_action_ratio': planning_before_action_ratio,
            'tasks_with_planning': sum(1 for count in planning_by_task.values() if count > 0)
        }
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """
        Analyze the quality of generated code.
        
        Returns:
            Dict: Code quality metrics
        """
        # Extract code blocks from the data
        code_blocks = []
        for event in self.data:
            if event.get('source') == 'agent' and event.get('action') in ['edit', 'run']:
                if 'args' in event:
                    # For edit actions
                    if 'new_str' in event['args'] and event['args']['new_str'] is not None:
                        code = event['args']['new_str']
                        code_blocks.append(code)
                    # For execute_ipython_cell or similar
                    elif 'code' in event['args'] and event['args']['code'] is not None:
                        code = event['args']['code']
                        code_blocks.append(code)
        
        # Calculate code metrics
        total_lines = sum(len(code.split('\n')) for code in code_blocks)
        avg_lines_per_block = total_lines / max(1, len(code_blocks))
        
        # Simple complexity metric: count of control structures
        control_structures = ['if', 'for', 'while', 'def', 'class']
        complexity_count = 0
        for code in code_blocks:
            for structure in control_structures:
                pattern = r'\b' + structure + r'\b'
                complexity_count += len(re.findall(pattern, code))
        
        avg_complexity = complexity_count / max(1, len(code_blocks))
        
        # Comment ratio (simplified)
        comment_lines = 0
        for code in code_blocks:
            lines = code.split('\n')
            comment_lines += sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('"""'))
        
        comment_ratio = comment_lines / max(1, total_lines)
        
        return {
            'total_code_blocks': len(code_blocks),
            'avg_lines_per_block': avg_lines_per_block,
            'avg_complexity': avg_complexity,
            'comment_ratio': comment_ratio,
            'total_lines': total_lines
        }
    
    def analyze_information_retrieval(self) -> Dict[str, Any]:
        """
        Analyze the efficiency of information retrieval.
        
        Returns:
            Dict: Information retrieval efficiency metrics
        """
        # Count information retrieval actions
        retrieval_actions = []
        for event in self.data:
            if (event.get('source') == 'agent' and event.get('action') == 'run' and 
                'args' in event and event['args'].get('name') in ['web_read', 'browser', 'str_replace_editor']):
                if event['args'].get('name') == 'str_replace_editor' and event['args'].get('command') == 'view':
                    retrieval_actions.append(event)
                elif event['args'].get('name') in ['web_read', 'browser']:
                    retrieval_actions.append(event)
        
        # Calculate retrieval ratio
        retrieval_ratio = len(retrieval_actions) / max(1, len(self.actions))
        
        # Calculate retrieval by task
        retrieval_by_task = defaultdict(int)
        for event in retrieval_actions:
            for task_id, events in self.events_by_task.items():
                if event in events:
                    retrieval_by_task[task_id] += 1
        
        avg_retrieval_per_task = sum(retrieval_by_task.values()) / max(1, len(self.tasks))
        
        # Calculate retrieval before action ratio
        retrieval_before_action = 0
        for task_id, events in self.events_by_task.items():
            action_events = [e for e in events if e.get('source') == 'agent' and 
                            e.get('action') not in ['run', 'system']]
            if action_events:
                first_action_idx = events.index(action_events[0])
                retrieval_before_first_action = sum(1 for e in events[:first_action_idx] 
                                                 if e in retrieval_actions)
                if retrieval_before_first_action > 0:
                    retrieval_before_action += 1
        
        retrieval_before_action_ratio = retrieval_before_action / max(1, len(self.tasks))
        
        return {
            'retrieval_ratio': retrieval_ratio,
            'avg_retrieval_per_task': avg_retrieval_per_task,
            'retrieval_before_action_ratio': retrieval_before_action_ratio,
            'tasks_with_retrieval': sum(1 for count in retrieval_by_task.values() if count > 0)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report with all metrics.
        
        Returns:
            Dict: Comprehensive report with all metrics
        """
        report = {
            'task_summary': {
                'total_tasks': len(self.tasks),
                'completed_tasks': sum(1 for status in self.task_completion_status.values() if status),
                'completion_rate': self.calculate_task_completion_rate()
            },
            'solution_quality': self.calculate_solution_quality(),
            'time_efficiency': self.calculate_time_efficiency(),
            'resource_usage': self.calculate_resource_usage(),
            'error_recovery': self.calculate_error_recovery(),
            'tool_selection': self.analyze_tool_selection(),
            'planning_quality': self.analyze_planning_quality(),
            'code_quality': self.analyze_code_quality(),
            'information_retrieval': self.analyze_information_retrieval()
        }
        
        return report
    
    def save_report(self, output_path: str):
        """
        Save the comprehensive report to a JSON file.
        
        Args:
            output_path: Path to save the report
        """
        report = self.generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {output_path}")
    
    def visualize_metrics(self, output_dir: str = None):
        """
        Generate visualizations for key metrics.
        
        Args:
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create output directory if it doesn't exist
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Get report data
            report = self.generate_comprehensive_report()
            
            # Set style
            sns.set(style="whitegrid")
            
            # 1. Task Completion Rate
            plt.figure(figsize=(10, 6))
            labels = ['Completed', 'Failed']
            sizes = [
                report['task_summary']['completion_rate'],
                100 - report['task_summary']['completion_rate']
            ]
            colors = ['#66b3ff', '#ff9999']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Task Completion Rate')
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'task_completion_rate.png'))
                plt.close()
            else:
                plt.show()
            
            # 2. Solution Quality Metrics
            plt.figure(figsize=(12, 6))
            quality_metrics = [
                report['solution_quality']['accuracy'],
                report['solution_quality']['robustness'],
                100 - report['solution_quality']['error_rate']
            ]
            plt.bar(['Accuracy', 'Robustness', 'Success Rate'], quality_metrics, color='#66b3ff')
            plt.ylim(0, 100)
            plt.title('Solution Quality Metrics')
            plt.ylabel('Percentage')
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'solution_quality.png'))
                plt.close()
            else:
                plt.show()
            
            # 3. Tool Usage Distribution
            plt.figure(figsize=(14, 8))
            tool_usage = report['resource_usage']['tool_usage_distribution']
            tools = list(tool_usage.keys())
            counts = list(tool_usage.values())
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            tools = [tools[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            plt.bar(tools, counts, color='#66b3ff')
            plt.xticks(rotation=45, ha='right')
            plt.title('Tool Usage Distribution')
            plt.ylabel('Count')
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'tool_usage.png'))
                plt.close()
            else:
                plt.show()
            
            # 4. Time Efficiency
            plt.figure(figsize=(10, 6))
            time_metrics = [
                report['time_efficiency']['avg_duration'],
                report['time_efficiency']['median_duration'],
                report['time_efficiency']['min_duration'],
                report['time_efficiency']['max_duration']
            ]
            plt.bar(['Average', 'Median', 'Minimum', 'Maximum'], time_metrics, color='#66b3ff')
            plt.title('Task Duration (seconds)')
            plt.ylabel('Seconds')
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'time_efficiency.png'))
                plt.close()
            else:
                plt.show()
            
            print("Visualizations generated successfully.")
            
        except ImportError:
            print("Visualization requires matplotlib and seaborn. Please install them with: pip install matplotlib seaborn")


def main():
    """
    Main function to demonstrate the usage of the OpenHandsDataProcessor.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Process OpenHands event stream data')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--output', '-o', type=str, default='report.json', help='Path to save the output report')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations')
    parser.add_argument('--viz-dir', type=str, default='visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Process the data
    processor = OpenHandsDataProcessor(data_path=args.input)
    
    # Generate and save the report
    processor.save_report(args.output)
    
    # Generate visualizations if requested
    if args.visualize:
        processor.visualize_metrics(args.viz_dir)
    
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