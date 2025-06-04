"""
Step-wise Agent Evaluation Framework

This module provides comprehensive evaluation of agent performance on a step-by-step basis,
analyzing efficiency, error recovery, task completion, and other key metrics.
"""

import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import LLM evaluation components
try:
    from llm_evaluator import (
        LLMEvaluator, EvaluationMetric, EvaluationContext, 
        EvaluationResult, create_llm_evaluator
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM evaluation module not available. LLM-based metrics will be skipped.")


class StepWiseEvaluator:
    """
    Comprehensive step-wise evaluation framework for agent performance analysis.
    """
    
    def __init__(self, data_path: str = None, data: List[Dict] = None, 
                 llm_config: Dict[str, Any] = None):
        """
        Initialize the evaluator with event stream data.
        
        Args:
            data_path: Path to the JSON file containing event stream data
            data: Direct list of event dictionaries
            llm_config: Configuration for LLM evaluation (provider, api_key, etc.)
        """
        if data_path:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        elif data:
            self.raw_data = data
        else:
            raise ValueError("Either data_path or data must be provided")
        
        # Filter data starting from ID 3 as requested
        self.data = [entry for entry in self.raw_data if entry.get('id', 0) >= 3]
        
        # Initialize containers
        self.tasks = []
        self.task_boundaries = {}
        self.step_metrics = []
        self.task_metrics = []
        self.error_events = []
        self.llm_evaluation_results = {}
        self.llm_evaluation_logs = []  # New: Store detailed LLM evaluation logs
        
        # Initialize LLM evaluator if available and configured
        self.llm_evaluator = None
        self.llm_config = llm_config or {}
        if LLM_AVAILABLE and self.llm_config:
            try:
                provider_type = self.llm_config.get('provider', 'mock')
                # Filter out the 'provider' key before passing kwargs
                provider_kwargs = {k: v for k, v in self.llm_config.items() if k != 'provider'}
                self.llm_evaluator = create_llm_evaluator(provider_type, **provider_kwargs)
            except Exception as e:
                print(f"Warning: Failed to initialize LLM evaluator: {e}")
        
        # Process the data
        self._identify_task_boundaries()
        self._extract_step_metrics()
        self._calculate_task_metrics()
    
    def _identify_task_boundaries(self):
        """Identify precise task boundaries from user messages and finish events automatically."""
        user_messages = []
        finish_events = []
        
        # Collect all user messages and finish events
        for i, entry in enumerate(self.data):
            if entry.get('source') == 'user' and entry.get('action') == 'message':
                user_messages.append((i, entry.get('id'), entry.get('message', '')))
            elif entry.get('action') == 'finish':
                finish_events.append((i, entry.get('id'), entry.get('args', {}).get('task_completed')))
        
        print(f"Found {len(user_messages)} user messages and {len(finish_events)} finish events")
        
        # Sort by ID to ensure proper order
        user_messages.sort(key=lambda x: x[1])
        finish_events.sort(key=lambda x: x[1])
        
        # Match user messages with finish events
        self.task_boundaries = {}
        
        for i, (msg_idx, msg_id, message) in enumerate(user_messages):
            task_num = i + 1
            start_id = msg_id
            
            # Find the corresponding finish event for this task
            end_id = None
            
            # Look for finish events that come after this user message
            # but before the next user message (if any)
            next_user_id = user_messages[i + 1][1] if i + 1 < len(user_messages) else float('inf')
            
            # Find finish events between current user message and next user message
            candidate_finish_events = [
                (finish_idx, finish_id, task_completed) 
                for finish_idx, finish_id, task_completed in finish_events
                if start_id < finish_id < next_user_id
            ]
            
            if candidate_finish_events:
                # Use the last finish event in this range (closest to the next task)
                end_id = candidate_finish_events[-1][1]
            else:
                # No finish event found, use the ID just before the next user message
                if i + 1 < len(user_messages):
                    end_id = user_messages[i + 1][1] - 1
                else:
                    # Last task, use the last entry in the data
                    end_id = self.data[-1].get('id')
            
            # Store task boundary
            self.task_boundaries[task_num] = {
                'start_id': start_id,
                'end_id': end_id,
                'message': message[:100] + '...' if len(message) > 100 else message
            }
            
            print(f"Task {task_num}: IDs {start_id}-{end_id} | {message[:50]}...")
        
        print(f"Successfully identified {len(self.task_boundaries)} tasks")
    
    def _extract_step_metrics(self):
        """Extract detailed metrics for each step."""
        for entry in self.data:
            entry_id = entry.get('id')
            timestamp = entry.get('timestamp')
            source = entry.get('source')
            action = entry.get('action')
            
            # Determine which task this step belongs to
            task_id = self._get_task_for_step(entry_id)
            
            # Extract LLM metrics
            llm_metrics = entry.get('llm_metrics', {})
            token_usage = llm_metrics.get('accumulated_token_usage', {})
            
            # Extract tool information
            tool_info = self._extract_tool_info(entry)
            
            # Check for errors
            is_error = self._is_error_step(entry)
            
            # Calculate step duration (approximate)
            step_duration = self._calculate_step_duration(entry, entry_id)
            
            step_metric = {
                'step_id': entry_id,
                'task_id': task_id,
                'timestamp': timestamp,
                'source': source,
                'action': action,
                'duration_seconds': step_duration,
                'is_error': is_error,
                'tool_used': tool_info.get('tool_name'),
                'tool_args': tool_info.get('tool_args'),
                'tokens_prompt': token_usage.get('prompt_tokens', 0),
                'tokens_completion': token_usage.get('completion_tokens', 0),
                'tokens_total': token_usage.get('per_turn_token', 0),
                'cost': llm_metrics.get('accumulated_cost', 0),
                'lines_of_code': self._count_lines_of_code(entry),
                'complexity_score': self._calculate_complexity_score(entry)
            }
            
            self.step_metrics.append(step_metric)
            
            if is_error:
                self.error_events.append(entry)
    
    def _get_task_for_step(self, step_id: int) -> Optional[int]:
        """Determine which task a step belongs to."""
        for task_id, boundaries in self.task_boundaries.items():
            if boundaries['start_id'] <= step_id <= boundaries['end_id']:
                return task_id
        return None
    
    def _extract_tool_info(self, entry: Dict) -> Dict:
        """Extract tool usage information from an entry."""
        tool_info = {'tool_name': None, 'tool_args': None}
        
        if entry.get('action') in ['edit', 'run']:
            # For edit/run actions
            if entry.get('action') == 'edit':
                tool_info['tool_name'] = 'str_replace_editor'
            elif entry.get('action') == 'run':
                tool_info['tool_name'] = 'execute_bash'
            
            tool_info['tool_args'] = entry.get('args', {})
        
        # Check tool_call_metadata for more detailed info
        metadata = entry.get('tool_call_metadata', {})
        if metadata:
            function_name = metadata.get('function_name')
            if function_name:
                tool_info['tool_name'] = function_name
        
        return tool_info
    
    def _is_error_step(self, entry: Dict) -> bool:
        """Check if a step contains an error."""
        # Check various fields for error indicators
        message = str(entry.get('message', '')).lower()
        observation = str(entry.get('observation', '')).lower()
        content = str(entry.get('content', '')).lower()
        
        error_keywords = ['error', 'failed', 'exception', 'traceback', 'syntax error']
        
        for keyword in error_keywords:
            if keyword in message or keyword in observation or keyword in content:
                return True
        
        # Check args for error indicators
        args = entry.get('args', {})
        if isinstance(args, dict):
            for value in args.values():
                if isinstance(value, str) and any(keyword in value.lower() for keyword in error_keywords):
                    return True
        
        return False
    
    def _calculate_step_duration(self, entry: Dict, entry_id: int) -> float:
        """Calculate approximate duration for a step."""
        try:
            current_time = datetime.fromisoformat(entry.get('timestamp').replace('Z', '+00:00'))
            
            # Find next entry
            next_entry = None
            for e in self.data:
                if e.get('id', 0) > entry_id:
                    next_entry = e
                    break
            
            if next_entry:
                next_time = datetime.fromisoformat(next_entry.get('timestamp').replace('Z', '+00:00'))
                return (next_time - current_time).total_seconds()
            
            return 0.0
        except:
            return 0.0
    
    def _count_lines_of_code(self, entry: Dict) -> int:
        """Count lines of code in an entry."""
        lines = 0
        args = entry.get('args', {})
        
        # Check different fields that might contain code
        code_fields = ['file_text', 'new_str', 'command']
        for field in code_fields:
            if field in args and args[field]:
                lines += len(str(args[field]).split('\n'))
        
        return lines
    
    def _calculate_complexity_score(self, entry: Dict) -> int:
        """Calculate a simple complexity score for code."""
        complexity = 0
        args = entry.get('args', {})
        
        # Get code content
        code_content = ""
        code_fields = ['file_text', 'new_str', 'command']
        for field in code_fields:
            if field in args and args[field]:
                code_content += str(args[field]) + "\n"
        
        if code_content:
            # Count control structures
            control_keywords = ['if', 'for', 'while', 'def', 'class', 'try', 'except']
            for keyword in control_keywords:
                complexity += len(re.findall(r'\b' + keyword + r'\b', code_content))
        
        return complexity
    
    def _calculate_task_metrics(self):
        """Calculate aggregated metrics for each task."""
        for task_id in self.task_boundaries.keys():
            task_steps = [step for step in self.step_metrics if step['task_id'] == task_id]
            
            if not task_steps:
                continue
            
            # Basic metrics
            total_steps = len(task_steps)
            total_duration = sum(step['duration_seconds'] for step in task_steps)
            avg_step_duration = total_duration / total_steps if total_steps > 0 else 0
            
            # Error metrics
            error_steps = [step for step in task_steps if step['is_error']]
            error_count = len(error_steps)
            error_rate = error_count / total_steps if total_steps > 0 else 0
            
            # LLM metrics
            total_tokens = sum(step['tokens_total'] for step in task_steps)
            total_cost = max(step['cost'] for step in task_steps) if task_steps else 0
            
            # Code metrics
            total_lines = sum(step['lines_of_code'] for step in task_steps)
            total_complexity = sum(step['complexity_score'] for step in task_steps)
            
            # Tool usage
            tools_used = [step['tool_used'] for step in task_steps if step['tool_used']]
            unique_tools = len(set(tools_used))
            
            # Action distribution
            actions = [step['action'] for step in task_steps if step['action']]
            action_dist = Counter(actions)
            
            # Task completion
            is_completed = self._is_task_completed(task_id)
            
            task_metric = {
                'task_id': task_id,
                'task_message': self.task_boundaries[task_id]['message'],
                'total_steps': total_steps,
                'total_duration_seconds': total_duration,
                'avg_step_duration': avg_step_duration,
                'error_count': error_count,
                'error_rate': error_rate,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'total_lines_of_code': total_lines,
                'total_complexity': total_complexity,
                'unique_tools_used': unique_tools,
                'action_distribution': dict(action_dist),
                'is_completed': is_completed,
                'efficiency_score': self._calculate_efficiency_score(task_steps),
                'quality_score': self._calculate_quality_score(task_steps)
            }
            
            self.task_metrics.append(task_metric)
    
    def _is_task_completed(self, task_id: int) -> bool:
        """Check if a task was completed successfully."""
        boundaries = self.task_boundaries[task_id]
        end_id = boundaries['end_id']
        
        # Find finish event near the end of the task
        for entry in self.data:
            if (entry.get('id') >= end_id - 5 and 
                entry.get('id') <= end_id + 5 and 
                entry.get('action') == 'finish'):
                return entry.get('args', {}).get('task_completed') == 'true'
        
        return False
    
    def _calculate_efficiency_score(self, task_steps: List[Dict]) -> float:
        """Calculate efficiency score for a task (0-100)."""
        if not task_steps:
            return 0.0
        
        # Factors: fewer steps, less time, fewer errors
        total_steps = len(task_steps)
        total_duration = sum(step['duration_seconds'] for step in task_steps)
        error_count = sum(1 for step in task_steps if step['is_error'])
        
        # Normalize and combine (lower is better for steps, duration, errors)
        step_score = max(0, 100 - total_steps * 2)  # Penalty for many steps
        time_score = max(0, 100 - total_duration / 10)  # Penalty for long duration
        error_score = max(0, 100 - error_count * 20)  # Penalty for errors
        
        return (step_score + time_score + error_score) / 3
    
    def _calculate_quality_score(self, task_steps: List[Dict]) -> float:
        """Calculate quality score for a task (0-100)."""
        if not task_steps:
            return 0.0
        
        # Factors: code complexity, tool usage diversity, successful completion
        total_complexity = sum(step['complexity_score'] for step in task_steps)
        tools_used = [step['tool_used'] for step in task_steps if step['tool_used']]
        unique_tools = len(set(tools_used))
        
        # Reward reasonable complexity and tool diversity
        complexity_score = min(100, total_complexity * 10)  # Reward complexity up to a point
        tool_diversity_score = min(100, unique_tools * 20)  # Reward tool diversity
        
        return (complexity_score + tool_diversity_score) / 2
    
    def get_efficiency_metrics(self) -> Dict[str, Any]:
        """Get efficiency-related metrics."""
        if not self.task_metrics:
            return {}
        
        total_tasks = len(self.task_metrics)
        
        return {
            'average_steps_per_task': np.mean([t['total_steps'] for t in self.task_metrics]),
            'average_time_per_task': np.mean([t['total_duration_seconds'] for t in self.task_metrics]),
            'average_time_per_step': np.mean([t['avg_step_duration'] for t in self.task_metrics]),
            'total_steps_all_tasks': sum(t['total_steps'] for t in self.task_metrics),
            'total_time_all_tasks': sum(t['total_duration_seconds'] for t in self.task_metrics),
            'efficiency_scores': [t['efficiency_score'] for t in self.task_metrics],
            'average_efficiency_score': np.mean([t['efficiency_score'] for t in self.task_metrics])
        }
    
    def get_error_recovery_metrics(self) -> Dict[str, Any]:
        """Get error recovery metrics."""
        if not self.task_metrics:
            return {}
        
        total_errors = sum(t['error_count'] for t in self.task_metrics)
        tasks_with_errors = sum(1 for t in self.task_metrics if t['error_count'] > 0)
        completed_tasks_with_errors = sum(1 for t in self.task_metrics 
                                        if t['error_count'] > 0 and t['is_completed'])
        
        recovery_rate = (completed_tasks_with_errors / tasks_with_errors * 100 
                        if tasks_with_errors > 0 else 0)
        
        return {
            'total_errors': total_errors,
            'average_errors_per_task': total_errors / len(self.task_metrics),
            'tasks_with_errors': tasks_with_errors,
            'error_recovery_rate': recovery_rate,
            'error_rates_per_task': [t['error_rate'] * 100 for t in self.task_metrics],
            'average_error_rate': np.mean([t['error_rate'] for t in self.task_metrics]) * 100
        }
    
    def get_task_completion_metrics(self) -> Dict[str, Any]:
        """Get task completion and LLM usage metrics."""
        if not self.task_metrics:
            return {}
        
        completed_tasks = sum(1 for t in self.task_metrics if t['is_completed'])
        completion_rate = completed_tasks / len(self.task_metrics) * 100
        
        total_tokens = sum(t['total_tokens'] for t in self.task_metrics)
        total_cost = sum(t['total_cost'] for t in self.task_metrics)
        
        return {
            'total_tasks': len(self.task_metrics),
            'completed_tasks': completed_tasks,
            'completion_rate': completion_rate,
            'total_tokens_used': total_tokens,
            'average_tokens_per_task': total_tokens / len(self.task_metrics),
            'total_cost': total_cost,
            'average_cost_per_task': total_cost / len(self.task_metrics),
            'tokens_per_task': [t['total_tokens'] for t in self.task_metrics],
            'cost_per_task': [t['total_cost'] for t in self.task_metrics]
        }
    
    def get_code_quality_metrics(self) -> Dict[str, Any]:
        """Get code quality metrics."""
        if not self.task_metrics:
            return {}
        
        total_lines = sum(t['total_lines_of_code'] for t in self.task_metrics)
        total_complexity = sum(t['total_complexity'] for t in self.task_metrics)
        
        return {
            'total_lines_of_code': total_lines,
            'average_lines_per_task': total_lines / len(self.task_metrics),
            'total_complexity': total_complexity,
            'average_complexity_per_task': total_complexity / len(self.task_metrics),
            'quality_scores': [t['quality_score'] for t in self.task_metrics],
            'average_quality_score': np.mean([t['quality_score'] for t in self.task_metrics]),
            'lines_per_task': [t['total_lines_of_code'] for t in self.task_metrics],
            'complexity_per_task': [t['total_complexity'] for t in self.task_metrics]
        }
    
    def get_tool_usage_metrics(self) -> Dict[str, Any]:
        """Get tool usage and selection metrics."""
        all_tools = []
        for step in self.step_metrics:
            if step['tool_used']:
                all_tools.append(step['tool_used'])
        
        tool_distribution = Counter(all_tools)
        unique_tools = len(set(all_tools))
        
        return {
            'total_tool_calls': len(all_tools),
            'unique_tools_used': unique_tools,
            'tool_distribution': dict(tool_distribution),
            'most_used_tools': dict(tool_distribution.most_common(5)),
            'average_tools_per_task': np.mean([t['unique_tools_used'] for t in self.task_metrics]),
            'tools_per_task': [t['unique_tools_used'] for t in self.task_metrics]
        }
    
    def _extract_code_for_task(self, task_id: int) -> str:
        """Extract all code generated for a specific task."""
        task_steps = [step for step in self.step_metrics if step['task_id'] == task_id]
        code_parts = []
        
        for step in task_steps:
            if step['tool_args'] and isinstance(step['tool_args'], dict):
                # Extract code from various fields
                for field in ['file_text', 'new_str', 'command']:
                    if field in step['tool_args'] and step['tool_args'][field]:
                        code_content = str(step['tool_args'][field])
                        if len(code_content.strip()) > 10:  # Only meaningful code
                            code_parts.append(code_content)
        
        return '\n\n'.join(code_parts)
    
    def _get_similar_tasks(self, current_task_id: int) -> List[Dict[str, Any]]:
        """Get similar tasks for comparison in LLM evaluation."""
        current_task = next((t for t in self.task_metrics if t['task_id'] == current_task_id), None)
        if not current_task:
            return []
        
        similar_tasks = []
        for task in self.task_metrics:
            if task['task_id'] != current_task_id:
                # Simple similarity based on task message keywords
                current_words = set(current_task['task_message'].lower().split())
                task_words = set(task['task_message'].lower().split())
                similarity = len(current_words & task_words) / len(current_words | task_words)
                
                if similarity > 0.3:  # Threshold for similarity
                    similar_tasks.append({
                        'task_id': task['task_id'],
                        'message': task['task_message'],
                        'steps': task['total_steps'],
                        'duration': task['total_duration_seconds'],
                        'tokens': task['total_tokens'],
                        'cost': task['total_cost'],
                        'completed': task['is_completed'],
                        'similarity_score': similarity
                    })
        
        return sorted(similar_tasks, key=lambda x: x['similarity_score'], reverse=True)[:3]
    
    async def run_llm_evaluation(self, selected_metrics: List[str] = None) -> Dict[str, Any]:
        """Run LLM-based evaluation for abstract metrics."""
        if not self.llm_evaluator:
            return {
                'error': 'LLM evaluator not available or not configured',
                'results': {}
            }
        
        # Default metrics to evaluate
        if selected_metrics is None:
            selected_metrics = [
                'step_wise_correctness',
                'tool_selection_efficiency', 
                'planning_quality',
                'code_quality',
                'consistency',
                'learning_adaptability'
            ]
        
        # Initialize LLM evaluation logs
        self.llm_evaluation_logs = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'llm_provider': self.llm_evaluator.provider.__class__.__name__,
                'llm_model': self.llm_evaluator.provider.model,
                'metrics_evaluated': selected_metrics,
                'total_tasks': len(self.task_boundaries)
            },
            'detailed_logs': []
        }
        
        evaluation_results = {}
        
        for task_id in self.task_boundaries.keys():
            task_boundary = self.task_boundaries[task_id]
            task_steps = [step for step in self.step_metrics if step['task_id'] == task_id]
            
            # Extract code for this task
            code_generated = self._extract_code_for_task(task_id)
            
            # Get similar tasks for comparison
            similar_tasks = self._get_similar_tasks(task_id)
            
            # Get errors for this task
            task_errors = [step for step in task_steps if step['is_error']]
            
            # Create evaluation context
            context = EvaluationContext(
                task_id=task_id,
                task_description=task_boundary['message'],
                steps=task_steps,
                code_generated=code_generated,
                tools_available=['str_replace_editor', 'execute_bash', 'think', 'browser'],
                errors_encountered=task_errors,
                similar_tasks=similar_tasks
            )
            
            # Create task log entry
            task_log = {
                'task_id': task_id,
                'task_description': task_boundary['message'],
                'task_context': {
                    'total_steps': len(task_steps),
                    'code_length': len(code_generated),
                    'error_count': len(task_errors),
                    'similar_tasks_count': len(similar_tasks)
                },
                'metric_evaluations': []
            }
            
            # Evaluate each selected metric
            task_results = {}
            for metric_name in selected_metrics:
                try:
                    metric = EvaluationMetric(metric_name)
                    
                    # Skip certain metrics if data is not available
                    if metric == EvaluationMetric.CODE_QUALITY and not code_generated:
                        continue
                    if metric in [EvaluationMetric.CONSISTENCY, EvaluationMetric.LEARNING_ADAPTABILITY] and not similar_tasks:
                        continue
                    
                    # Get the prompt that will be sent to the LLM
                    prompt = self.llm_evaluator._get_prompt_for_metric(metric, context)
                    
                    # Log the evaluation start
                    evaluation_start = datetime.now()
                    
                    # Perform the evaluation
                    result = await self.llm_evaluator.evaluate_metric(metric, context)
                    
                    # Log the evaluation end
                    evaluation_end = datetime.now()
                    evaluation_duration = (evaluation_end - evaluation_start).total_seconds()
                    
                    # Create detailed log entry for this metric evaluation
                    metric_log = {
                        'metric': metric_name,
                        'timestamp': evaluation_start.isoformat(),
                        'duration_seconds': evaluation_duration,
                        'prompt': prompt,
                        'response': {
                            'score': result.score,
                            'reasoning': result.reasoning,
                            'confidence': result.confidence,
                            'details': result.details
                        },
                        'raw_llm_response': getattr(result, 'raw_response', None),
                        'status': 'success' if result.score > 0 else 'partial_success'
                    }
                    
                    task_log['metric_evaluations'].append(metric_log)
                    
                    task_results[metric_name] = {
                        'score': result.score,
                        'reasoning': result.reasoning,
                        'confidence': result.confidence,
                        'details': result.details
                    }
                    
                except Exception as e:
                    print(f"Error evaluating {metric_name} for task {task_id}: {e}")
                    
                    # Log the error
                    error_log = {
                        'metric': metric_name,
                        'timestamp': datetime.now().isoformat(),
                        'duration_seconds': 0,
                        'prompt': getattr(self.llm_evaluator, '_get_prompt_for_metric', lambda m, c: 'Prompt generation failed')(metric, context) if hasattr(self.llm_evaluator, '_get_prompt_for_metric') else 'Prompt unavailable',
                        'response': None,
                        'error': str(e),
                        'status': 'error'
                    }
                    
                    task_log['metric_evaluations'].append(error_log)
                    
                    task_results[metric_name] = {
                        'score': 5.0,
                        'reasoning': f"Evaluation failed: {str(e)}",
                        'confidence': 0.0,
                        'details': {}
                    }
            
            # Add task log to detailed logs
            self.llm_evaluation_logs['detailed_logs'].append(task_log)
            
            evaluation_results[f'task_{task_id}'] = task_results
        
        # Calculate aggregate scores
        aggregate_scores = {}
        for metric_name in selected_metrics:
            scores = []
            confidences = []
            
            for task_results in evaluation_results.values():
                if metric_name in task_results:
                    scores.append(task_results[metric_name]['score'])
                    confidences.append(task_results[metric_name]['confidence'])
            
            if scores:
                aggregate_scores[metric_name] = {
                    'average_score': np.mean(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'std_score': np.std(scores),
                    'average_confidence': np.mean(confidences),
                    'scores_per_task': scores
                }
        
        # Add summary statistics to logs
        self.llm_evaluation_logs['evaluation_summary'] = {
            'total_evaluations': sum(len(task_log['metric_evaluations']) for task_log in self.llm_evaluation_logs['detailed_logs']),
            'successful_evaluations': sum(1 for task_log in self.llm_evaluation_logs['detailed_logs'] 
                                        for eval_log in task_log['metric_evaluations'] 
                                        if eval_log['status'] == 'success'),
            'failed_evaluations': sum(1 for task_log in self.llm_evaluation_logs['detailed_logs'] 
                                    for eval_log in task_log['metric_evaluations'] 
                                    if eval_log['status'] == 'error'),
            'average_duration_per_evaluation': np.mean([eval_log['duration_seconds'] 
                                                      for task_log in self.llm_evaluation_logs['detailed_logs'] 
                                                      for eval_log in task_log['metric_evaluations']
                                                      if eval_log['status'] != 'error']),
            'aggregate_scores': aggregate_scores
        }
        
        return {
            'task_results': evaluation_results,
            'aggregate_scores': aggregate_scores,
            'evaluation_summary': {
                'total_tasks_evaluated': len(evaluation_results),
                'metrics_evaluated': selected_metrics,
                'llm_provider': self.llm_evaluator.provider.__class__.__name__,
                'llm_model': self.llm_evaluator.provider.model
            }
        }
    
    def get_llm_evaluation_metrics(self) -> Dict[str, Any]:
        """Get LLM evaluation results if available."""
        if not self.llm_evaluation_results:
            return {
                'available': False,
                'message': 'LLM evaluation not run or not available'
            }
        
        return {
            'available': True,
            'results': self.llm_evaluation_results
        }
    
    def generate_comprehensive_report(self, include_llm_evaluation: bool = False) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        report = {
            'task_boundaries': self.task_boundaries,
            'efficiency_metrics': self.get_efficiency_metrics(),
            'error_recovery_metrics': self.get_error_recovery_metrics(),
            'task_completion_metrics': self.get_task_completion_metrics(),
            'code_quality_metrics': self.get_code_quality_metrics(),
            'tool_usage_metrics': self.get_tool_usage_metrics(),
            'detailed_task_metrics': self.task_metrics,
            'detailed_step_metrics': self.step_metrics
        }
        
        # Add LLM evaluation results if requested and available
        if include_llm_evaluation:
            report['llm_evaluation_metrics'] = self.get_llm_evaluation_metrics()
        
        return report
    
    def save_report(self, output_path: str):
        """Save the comprehensive report to a JSON file."""
        report = self.generate_comprehensive_report()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def clean_for_json(data):
            if isinstance(data, dict):
                return {k: clean_for_json(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_for_json(item) for item in data]
            else:
                return convert_numpy(data)
        
        clean_report = clean_for_json(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive report saved to {output_path}")
    
    def create_visualizations(self, output_dir: str = 'visualizations'):
        """Create comprehensive visualizations of the evaluation metrics."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Get metrics
        efficiency = self.get_efficiency_metrics()
        error_recovery = self.get_error_recovery_metrics()
        completion = self.get_task_completion_metrics()
        quality = self.get_code_quality_metrics()
        tools = self.get_tool_usage_metrics()
        
        # 1. Task Completion Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Completion rate pie chart
        completed = completion['completed_tasks']
        failed = completion['total_tasks'] - completed
        ax1.pie([completed, failed], labels=['Completed', 'Failed'], 
                autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
        ax1.set_title('Task Completion Rate')
        
        # Steps per task
        task_ids = list(range(1, len(self.task_metrics) + 1))
        steps_per_task = [t['total_steps'] for t in self.task_metrics]
        ax2.bar(task_ids, steps_per_task, color='#3498db')
        ax2.set_title('Steps per Task')
        ax2.set_xlabel('Task ID')
        ax2.set_ylabel('Number of Steps')
        
        # Time per task
        time_per_task = [t['total_duration_seconds'] for t in self.task_metrics]
        ax3.bar(task_ids, time_per_task, color='#9b59b6')
        ax3.set_title('Time per Task (seconds)')
        ax3.set_xlabel('Task ID')
        ax3.set_ylabel('Duration (seconds)')
        
        # Error rate per task
        error_rates = [t['error_rate'] * 100 for t in self.task_metrics]
        ax4.bar(task_ids, error_rates, color='#e67e22')
        ax4.set_title('Error Rate per Task (%)')
        ax4.set_xlabel('Task ID')
        ax4.set_ylabel('Error Rate (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'task_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Efficiency Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Efficiency scores
        efficiency_scores = [t['efficiency_score'] for t in self.task_metrics]
        ax1.plot(task_ids, efficiency_scores, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        ax1.set_title('Efficiency Score per Task')
        ax1.set_xlabel('Task ID')
        ax1.set_ylabel('Efficiency Score (0-100)')
        ax1.grid(True, alpha=0.3)
        
        # Quality scores
        quality_scores = [t['quality_score'] for t in self.task_metrics]
        ax2.plot(task_ids, quality_scores, marker='s', linewidth=2, markersize=8, color='#f39c12')
        ax2.set_title('Quality Score per Task')
        ax2.set_xlabel('Task ID')
        ax2.set_ylabel('Quality Score (0-100)')
        ax2.grid(True, alpha=0.3)
        
        # Lines of code per task
        lines_per_task = [t['total_lines_of_code'] for t in self.task_metrics]
        ax3.bar(task_ids, lines_per_task, color='#8e44ad')
        ax3.set_title('Lines of Code per Task')
        ax3.set_xlabel('Task ID')
        ax3.set_ylabel('Lines of Code')
        
        # Complexity per task
        complexity_per_task = [t['total_complexity'] for t in self.task_metrics]
        ax4.bar(task_ids, complexity_per_task, color='#34495e')
        ax4.set_title('Code Complexity per Task')
        ax4.set_xlabel('Task ID')
        ax4.set_ylabel('Complexity Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Tool Usage Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tool distribution
        tool_dist = tools['tool_distribution']
        tools_list = list(tool_dist.keys())
        counts = list(tool_dist.values())
        
        ax1.bar(tools_list, counts, color='#16a085')
        ax1.set_title('Tool Usage Distribution')
        ax1.set_xlabel('Tools')
        ax1.set_ylabel('Usage Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Tools per task
        tools_per_task = [t['unique_tools_used'] for t in self.task_metrics]
        ax2.bar(task_ids, tools_per_task, color='#27ae60')
        ax2.set_title('Unique Tools Used per Task')
        ax2.set_xlabel('Task ID')
        ax2.set_ylabel('Number of Unique Tools')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tool_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. LLM Usage Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tokens per task
        tokens_per_task = completion['tokens_per_task']
        ax1.bar(task_ids, tokens_per_task, color='#e74c3c')
        ax1.set_title('Token Usage per Task')
        ax1.set_xlabel('Task ID')
        ax1.set_ylabel('Total Tokens')
        
        # Cost per task
        cost_per_task = completion['cost_per_task']
        ax2.bar(task_ids, cost_per_task, color='#f39c12')
        ax2.set_title('Cost per Task')
        ax2.set_xlabel('Task ID')
        ax2.set_ylabel('Cost ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'llm_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def print_summary(self):
        """Print a comprehensive summary of the evaluation."""
        report = self.generate_comprehensive_report()
        
        print("=" * 60)
        print("STEP-WISE AGENT EVALUATION SUMMARY")
        print("=" * 60)
        
        # Task boundaries
        print(f"\nTASK BOUNDARIES:")
        for task_id, boundary in report['task_boundaries'].items():
            print(f"Task {task_id}: IDs {boundary['start_id']}-{boundary['end_id']}")
            print(f"  Message: {boundary['message']}")
        
        # Efficiency metrics
        eff = report['efficiency_metrics']
        print(f"\nEFFICIENCY METRICS:")
        print(f"  Average steps per task: {eff['average_steps_per_task']:.1f}")
        print(f"  Average time per task: {eff['average_time_per_task']:.1f} seconds")
        print(f"  Average time per step: {eff['average_time_per_step']:.2f} seconds")
        print(f"  Average efficiency score: {eff['average_efficiency_score']:.1f}/100")
        
        # Error recovery
        err = report['error_recovery_metrics']
        print(f"\nERROR RECOVERY METRICS:")
        print(f"  Total errors: {err['total_errors']}")
        print(f"  Average errors per task: {err['average_errors_per_task']:.1f}")
        print(f"  Error recovery rate: {err['error_recovery_rate']:.1f}%")
        print(f"  Average error rate: {err['average_error_rate']:.1f}%")
        
        # Task completion
        comp = report['task_completion_metrics']
        print(f"\nTASK COMPLETION METRICS:")
        print(f"  Total tasks: {comp['total_tasks']}")
        print(f"  Completed tasks: {comp['completed_tasks']}")
        print(f"  Completion rate: {comp['completion_rate']:.1f}%")
        print(f"  Total tokens used: {comp['total_tokens_used']:,}")
        print(f"  Total cost: ${comp['total_cost']:.4f}")
        
        # Code quality
        qual = report['code_quality_metrics']
        print(f"\nCODE QUALITY METRICS:")
        print(f"  Total lines of code: {qual['total_lines_of_code']}")
        print(f"  Average complexity per task: {qual['average_complexity_per_task']:.1f}")
        print(f"  Average quality score: {qual['average_quality_score']:.1f}/100")
        
        # Tool usage
        tools = report['tool_usage_metrics']
        print(f"\nTOOL USAGE METRICS:")
        print(f"  Total tool calls: {tools['total_tool_calls']}")
        print(f"  Unique tools used: {tools['unique_tools_used']}")
        print(f"  Most used tools: {list(tools['most_used_tools'].keys())[:3]}")


def main():
    """Main function to run the step-wise evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Step-wise Agent Performance Evaluation')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='Path to the input JSON file')
    parser.add_argument('--output', '-o', type=str, default='step_wise_report.json', 
                       help='Path to save the output report')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Generate visualizations')
    parser.add_argument('--viz-dir', type=str, default='visualizations', 
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = StepWiseEvaluator(data_path=args.input)
    
    # Generate and save report
    evaluator.save_report(args.output)
    
    # Create visualizations if requested
    if args.visualize:
        evaluator.create_visualizations(args.viz_dir)
    
    # Print summary
    evaluator.print_summary()


if __name__ == "__main__":
    main() 