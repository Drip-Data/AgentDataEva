#!/usr/bin/env python3
"""
Runner script for Step-wise Agent Evaluation

This script runs the comprehensive evaluation framework on the agent event stream data,
generates detailed reports, visualizations, and provides summary statistics.
"""

import os
import sys
import time
import argparse
import asyncio
from datetime import datetime
from io import StringIO
import contextlib
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from step_wise_evaluator import StepWiseEvaluator


class EvaluationLogger:
    """Logger to capture terminal output and generate markdown reports."""
    
    def __init__(self):
        self.logs = []
        self.start_time = None
        self.end_time = None
        
    def log(self, message, section=None):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.logs.append({
            'timestamp': timestamp,
            'message': message,
            'section': section
        })
        print(message)  # Still print to terminal
    
    def start_evaluation(self):
        """Mark the start of evaluation."""
        self.start_time = datetime.now()
        
    def end_evaluation(self):
        """Mark the end of evaluation."""
        self.end_time = datetime.now()
    
    def generate_markdown_report(self, output_path: str, args, evaluator, processing_time: float):
        """Generate a comprehensive markdown report."""
        report_lines = []
        
        # Header
        report_lines.append("# Step-wise Agent Evaluation Report")
        report_lines.append("")
        report_lines.append(f"**Generated on:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Processing time:** {processing_time:.2f} seconds")
        report_lines.append("")
        
        # Configuration
        report_lines.append("## Configuration")
        report_lines.append("")
        report_lines.append(f"- **Input file:** `{args.input}`")
        report_lines.append(f"- **Output directory:** `{args.output_dir}`")
        report_lines.append(f"- **LLM evaluation:** {'Enabled' if args.llm_eval else 'Disabled'}")
        if args.llm_eval:
            report_lines.append(f"- **LLM provider:** {args.llm_provider}")
            report_lines.append(f"- **LLM model:** {args.llm_model or 'Default'}")
        report_lines.append(f"- **Visualizations:** {'Enabled' if not args.no_viz else 'Disabled'}")
        report_lines.append("")
        
        # Data Overview
        report_lines.append("## Data Overview")
        report_lines.append("")
        report_lines.append(f"- **Total events processed:** {len(evaluator.data)}")
        report_lines.append(f"- **Tasks identified:** {len(evaluator.task_boundaries)}")
        report_lines.append(f"- **Step metrics extracted:** {len(evaluator.step_metrics)}")
        report_lines.append("")
        
        # Task Boundaries
        report_lines.append("## Task Boundaries")
        report_lines.append("")
        for task_id, boundary in evaluator.task_boundaries.items():
            report_lines.append(f"### Task {task_id}")
            report_lines.append(f"- **ID Range:** {boundary['start_id']}-{boundary['end_id']}")
            report_lines.append(f"- **Message:** {boundary['message']}")
            report_lines.append("")
        
        # Performance Metrics
        efficiency = evaluator.get_efficiency_metrics()
        error_recovery = evaluator.get_error_recovery_metrics()
        completion = evaluator.get_task_completion_metrics()
        quality = evaluator.get_code_quality_metrics()
        tools = evaluator.get_tool_usage_metrics()
        
        report_lines.append("## Performance Metrics")
        report_lines.append("")
        
        # Efficiency
        report_lines.append("### Efficiency Metrics")
        report_lines.append("")
        report_lines.append(f"- **Average steps per task:** {efficiency['average_steps_per_task']:.1f}")
        report_lines.append(f"- **Average time per task:** {efficiency['average_time_per_task']:.1f} seconds")
        report_lines.append(f"- **Average time per step:** {efficiency['average_time_per_step']:.2f} seconds")
        report_lines.append(f"- **Average efficiency score:** {efficiency['average_efficiency_score']:.1f}/100")
        report_lines.append("")
        
        # Error Recovery
        report_lines.append("### Error Recovery Metrics")
        report_lines.append("")
        report_lines.append(f"- **Total errors:** {error_recovery['total_errors']}")
        report_lines.append(f"- **Average errors per task:** {error_recovery['average_errors_per_task']:.1f}")
        report_lines.append(f"- **Error recovery rate:** {error_recovery['error_recovery_rate']:.1f}%")
        report_lines.append(f"- **Average error rate:** {error_recovery['average_error_rate']:.1f}%")
        report_lines.append("")
        
        # Task Completion
        report_lines.append("### Task Completion Metrics")
        report_lines.append("")
        report_lines.append(f"- **Total tasks:** {completion['total_tasks']}")
        report_lines.append(f"- **Completed tasks:** {completion['completed_tasks']}")
        report_lines.append(f"- **Completion rate:** {completion['completion_rate']:.1f}%")
        report_lines.append(f"- **Total tokens used:** {completion['total_tokens_used']:,}")
        report_lines.append(f"- **Total cost:** ${completion['total_cost']:.4f}")
        report_lines.append("")
        
        # Code Quality
        report_lines.append("### Code Quality Metrics")
        report_lines.append("")
        report_lines.append(f"- **Total lines of code:** {quality['total_lines_of_code']}")
        report_lines.append(f"- **Average complexity per task:** {quality['average_complexity_per_task']:.1f}")
        report_lines.append(f"- **Average quality score:** {quality['average_quality_score']:.1f}/100")
        report_lines.append("")
        
        # Tool Usage
        report_lines.append("### Tool Usage Metrics")
        report_lines.append("")
        report_lines.append(f"- **Total tool calls:** {tools['total_tool_calls']}")
        report_lines.append(f"- **Unique tools used:** {tools['unique_tools_used']}")
        report_lines.append(f"- **Most used tools:** {list(tools['most_used_tools'].keys())[:3]}")
        report_lines.append("")
        
        # LLM Evaluation Results
        if hasattr(evaluator, 'llm_evaluation_results') and evaluator.llm_evaluation_results:
            llm_results = evaluator.llm_evaluation_results
            if 'aggregate_scores' in llm_results:
                report_lines.append("## LLM Evaluation Results")
                report_lines.append("")
                report_lines.append(f"- **LLM Provider:** {llm_results['evaluation_summary'].get('llm_provider', 'Unknown')}")
                report_lines.append(f"- **Model:** {llm_results['evaluation_summary'].get('llm_model', 'Unknown')}")
                report_lines.append(f"- **Tasks evaluated:** {llm_results['evaluation_summary'].get('total_tasks_evaluated', 0)}")
                report_lines.append("")
                
                report_lines.append("### Aggregate LLM Scores (1-10 scale)")
                report_lines.append("")
                for metric, scores in llm_results['aggregate_scores'].items():
                    metric_name = metric.replace('_', ' ').title()
                    report_lines.append(f"#### {metric_name}")
                    report_lines.append(f"- **Average:** {scores['average_score']:.2f}")
                    report_lines.append(f"- **Range:** {scores['min_score']:.2f} - {scores['max_score']:.2f}")
                    report_lines.append(f"- **Confidence:** {scores['average_confidence']:.2f}")
        
        # Detailed Task Results
        report_lines.append("## Detailed Task Results")
        report_lines.append("")
        
        for task_metric in evaluator.task_metrics:
            task_id = task_metric['task_id']
            report_lines.append(f"### Task {task_id}")
            report_lines.append("")
            report_lines.append(f"- **Message:** {task_metric['task_message']}")
            report_lines.append(f"- **Steps:** {task_metric['total_steps']}")
            report_lines.append(f"- **Duration:** {task_metric['total_duration_seconds']:.1f} seconds")
            report_lines.append(f"- **Completed:** {'Yes' if task_metric['is_completed'] else 'No'}")
            report_lines.append(f"- **Errors:** {task_metric['error_count']} ({task_metric['error_rate']*100:.1f}%)")
            report_lines.append(f"- **Tokens:** {task_metric['total_tokens']:,}")
            report_lines.append(f"- **Cost:** ${task_metric['total_cost']:.4f}")
            report_lines.append(f"- **Lines of code:** {task_metric['total_lines_of_code']}")
            report_lines.append(f"- **Complexity:** {task_metric['total_complexity']}")
            report_lines.append(f"- **Tools used:** {task_metric['unique_tools_used']}")
            report_lines.append(f"- **Efficiency score:** {task_metric['efficiency_score']:.1f}/100")
            report_lines.append(f"- **Quality score:** {task_metric['quality_score']:.1f}/100")
            
            # LLM evaluation results for this task
            if hasattr(evaluator, 'llm_evaluation_results') and evaluator.llm_evaluation_results:
                task_key = f'task_{task_id}'
                if 'task_results' in evaluator.llm_evaluation_results and task_key in evaluator.llm_evaluation_results['task_results']:
                    llm_task_results = evaluator.llm_evaluation_results['task_results'][task_key]
                    report_lines.append("")
                    report_lines.append("#### LLM Evaluation Scores")
                    for metric, result in llm_task_results.items():
                        metric_name = metric.replace('_', ' ').title()
                        report_lines.append(f"- **{metric_name}:** {result['score']:.1f}/10 (confidence: {result['confidence']:.2f})")
            
            report_lines.append("")
        
        # Processing Log
        report_lines.append("## Processing Log")
        report_lines.append("")
        report_lines.append("```")
        for log_entry in self.logs:
            report_lines.append(f"[{log_entry['timestamp']}] {log_entry['message']}")
        report_lines.append("```")
        report_lines.append("")
        
        # Generated Files
        report_lines.append("## Generated Files")
        report_lines.append("")
        report_lines.append("- `evaluation_report.json` - Comprehensive evaluation data")
        report_lines.append("- `evaluation_summary.md` - This markdown summary")
        if hasattr(evaluator, 'llm_evaluation_results') and evaluator.llm_evaluation_results:
            report_lines.append("- `llm_evaluation_report.json` - Detailed LLM evaluation logs")
        if not args.no_viz:
            report_lines.append("- `visualizations/` - Performance charts and graphs")
            report_lines.append("  - `task_overview.png`")
            report_lines.append("  - `efficiency_analysis.png`")
            report_lines.append("  - `tool_usage.png`")
            report_lines.append("  - `llm_usage.png`")
        
        # Save markdown report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive step-wise evaluation of agent performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py                           # Run with default settings
  python run_evaluation.py --no-viz                  # Skip visualizations
  python run_evaluation.py --output-dir results/     # Custom output directory
  python run_evaluation.py --input data/custom.json # Custom input file
  
  # LLM Evaluation Examples:
  python run_evaluation.py --llm-eval --llm-provider mock                    # Mock LLM evaluation
  python run_evaluation.py --llm-eval --llm-provider openai --api-key sk-... # OpenAI evaluation
  python run_evaluation.py --llm-eval --llm-provider anthropic              # Anthropic evaluation
        """
    )
    
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        default='data/testv1.json',
        help='Path to the input JSON file (default: data/testv1.json)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress detailed output'
    )
    
    parser.add_argument(
        '--detailed-tasks',
        action='store_true',
        help='Print detailed metrics for each task'
    )
    
    # LLM Evaluation arguments
    parser.add_argument(
        '--llm-eval',
        action='store_true',
        help='Enable LLM-based evaluation for qualitative metrics'
    )
    
    parser.add_argument(
        '--llm-provider',
        choices=['openai', 'anthropic', 'gemini', 'deepseek', 'mock'],
        default='mock',
        help='LLM provider to use for evaluation'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for the LLM provider'
    )
    
    parser.add_argument(
        '--llm-model',
        type=str,
        help='Specific model to use (e.g., gpt-4, claude-3-sonnet-20240229, gemini-1.5-pro)'
    )
    
    parser.add_argument(
        '--llm-metrics',
        nargs='+',
        choices=[
            'consistency', 'step_wise_correctness', 'tool_selection_efficiency',
            'planning_quality', 'information_retrieval_efficiency', 'code_quality',
            'learning_adaptability'
        ],
        help='Specific LLM metrics to evaluate (default: all available)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = EvaluationLogger()
    logger.start_evaluation()
    
    if not args.quiet:
        logger.log("=" * 70)
        logger.log("STEP-WISE AGENT EVALUATION FRAMEWORK")
        logger.log("=" * 70)
        logger.log(f"Input file: {args.input}")
        logger.log(f"Output directory: {args.output_dir}")
        logger.log(f"LLM evaluation: {'Enabled' if args.llm_eval else 'Disabled'}")
        if args.llm_eval:
            logger.log(f"LLM provider: {args.llm_provider}")
        logger.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("")
    
    # Start evaluation
    start_time = time.time()
    
    try:
        # Prepare LLM configuration
        llm_config = None
        if args.llm_eval:
            llm_config = {
                'provider': args.llm_provider
            }
            
            if args.api_key:
                llm_config['api_key'] = args.api_key
            
            if args.llm_model:
                llm_config['model'] = args.llm_model
        
        if not args.quiet:
            logger.log("Loading and processing data...")
        
        # Create evaluator
        evaluator = StepWiseEvaluator(data_path=args.input, llm_config=llm_config)
        
        if not args.quiet:
            logger.log(f"✓ Loaded {len(evaluator.data)} events")
            logger.log(f"✓ Identified {len(evaluator.task_boundaries)} tasks")
            logger.log(f"✓ Extracted {len(evaluator.step_metrics)} step metrics")
            logger.log("")
        
        # Run LLM evaluation if enabled
        if args.llm_eval and evaluator.llm_evaluator:
            if not args.quiet:
                logger.log("Running LLM-based evaluation...")
            
            # Run async LLM evaluation
            llm_results = asyncio.run(evaluator.run_llm_evaluation(args.llm_metrics))
            evaluator.llm_evaluation_results = llm_results
            
            if not args.quiet:
                if 'error' in llm_results:
                    logger.log(f"⚠ LLM evaluation error: {llm_results['error']}")
                else:
                    logger.log(f"✓ LLM evaluation completed for {llm_results['evaluation_summary']['total_tasks_evaluated']} tasks")
                    logger.log(f"✓ Evaluated metrics: {', '.join(llm_results['evaluation_summary']['metrics_evaluated'])}")
                logger.log("")
            
            # Save LLM evaluation report if available
            if hasattr(evaluator, 'llm_evaluation_logs') and evaluator.llm_evaluation_logs:
                llm_report_path = os.path.join(args.output_dir, 'llm_evaluation_report.json')
                with open(llm_report_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluator.llm_evaluation_logs, f, indent=2, ensure_ascii=False)
                if not args.quiet:
                    logger.log(f"✓ LLM evaluation logs saved to: {llm_report_path}")
        
        # Generate comprehensive report
        if not args.quiet:
            logger.log("Generating comprehensive report...")
        
        report_path = os.path.join(args.output_dir, 'evaluation_report.json')
        evaluator.save_report(report_path)
        
        if not args.quiet:
            logger.log(f"✓ Report saved to: {report_path}")
        
        # Generate visualizations
        if not args.no_viz:
            if not args.quiet:
                logger.log("Creating visualizations...")
            
            viz_dir = os.path.join(args.output_dir, 'visualizations')
            evaluator.create_visualizations(viz_dir)
            
            if not args.quiet:
                logger.log(f"✓ Visualizations saved to: {viz_dir}/")
        
        # Print summary
        if not args.quiet:
            logger.log("\n" + "=" * 70)
            logger.log("EVALUATION SUMMARY")
            logger.log("=" * 70)
        
        # Capture summary output
        if not args.quiet:
            # Use StringIO to capture print_summary output
            summary_buffer = StringIO()
            with contextlib.redirect_stdout(summary_buffer):
                evaluator.print_summary()
            summary_output = summary_buffer.getvalue()
            
            # Log each line
            for line in summary_output.split('\n'):
                if line.strip():
                    logger.log(line)
        else:
            evaluator.print_summary()
        
        # Print LLM evaluation summary if available
        if args.llm_eval and evaluator.llm_evaluation_results and 'aggregate_scores' in evaluator.llm_evaluation_results:
            if not args.quiet:
                # Capture LLM summary
                llm_buffer = StringIO()
                with contextlib.redirect_stdout(llm_buffer):
                    print_llm_evaluation_summary(evaluator.llm_evaluation_results)
                llm_output = llm_buffer.getvalue()
                
                for line in llm_output.split('\n'):
                    if line.strip():
                        logger.log(line)
            else:
                print_llm_evaluation_summary(evaluator.llm_evaluation_results)
        
        # Print detailed task metrics if requested
        if args.detailed_tasks:
            if not args.quiet:
                # Capture detailed metrics
                detailed_buffer = StringIO()
                with contextlib.redirect_stdout(detailed_buffer):
                    print_detailed_task_metrics(evaluator)
                detailed_output = detailed_buffer.getvalue()
                
                for line in detailed_output.split('\n'):
                    if line.strip():
                        logger.log(line)
            else:
                print_detailed_task_metrics(evaluator)
        
        # Print performance statistics
        end_time = time.time()
        processing_time = end_time - start_time
        logger.end_evaluation()
        
        if not args.quiet:
            logger.log("\n" + "=" * 70)
            logger.log("PROCESSING STATISTICS")
            logger.log("=" * 70)
            logger.log(f"Processing time: {processing_time:.2f} seconds")
            logger.log(f"Events per second: {len(evaluator.data) / processing_time:.1f}")
            logger.log(f"Output files generated:")
            logger.log(f"  - {report_path}")
            if not args.no_viz:
                logger.log(f"  - {viz_dir}/task_overview.png")
                logger.log(f"  - {viz_dir}/efficiency_analysis.png")
                logger.log(f"  - {viz_dir}/tool_usage.png")
                logger.log(f"  - {viz_dir}/llm_usage.png")
        
        # Generate markdown report
        markdown_path = os.path.join(args.output_dir, 'evaluation_summary.md')
        logger.generate_markdown_report(markdown_path, args, evaluator, processing_time)
        
        if not args.quiet:
            logger.log(f"✓ Markdown summary saved to: {markdown_path}")
        
        logger.log(f"\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        logger.log(f"Error during evaluation: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def print_llm_evaluation_summary(llm_results: dict):
    """Print summary of LLM evaluation results."""
    print("\n" + "=" * 70)
    print("LLM EVALUATION SUMMARY")
    print("=" * 70)
    
    if 'evaluation_summary' in llm_results:
        summary = llm_results['evaluation_summary']
        print(f"LLM Provider: {summary.get('llm_provider', 'Unknown')}")
        print(f"Model: {summary.get('llm_model', 'Unknown')}")
        print(f"Tasks evaluated: {summary.get('total_tasks_evaluated', 0)}")
        print(f"Metrics evaluated: {', '.join(summary.get('metrics_evaluated', []))}")
    
    if 'aggregate_scores' in llm_results:
        print(f"\nAGGREGATE LLM SCORES (1-10 scale):")
        for metric, scores in llm_results['aggregate_scores'].items():
            print(f"  {metric.replace('_', ' ').title()}:")
            print(f"    Average: {scores['average_score']:.2f}")
            print(f"    Range: {scores['min_score']:.2f} - {scores['max_score']:.2f}")
            print(f"    Confidence: {scores['average_confidence']:.2f}")


def print_detailed_task_metrics(evaluator):
    """Print detailed metrics for each task."""
    print("\n" + "=" * 70)
    print("DETAILED TASK METRICS")
    print("=" * 70)
    
    for task_metric in evaluator.task_metrics:
        task_id = task_metric['task_id']
        print(f"\nTask {task_id}:")
        print(f"  Message: {task_metric['task_message']}")
        print(f"  Steps: {task_metric['total_steps']}")
        print(f"  Duration: {task_metric['total_duration_seconds']:.1f} seconds")
        print(f"  Completed: {'Yes' if task_metric['is_completed'] else 'No'}")
        print(f"  Errors: {task_metric['error_count']} ({task_metric['error_rate']*100:.1f}%)")
        print(f"  Tokens: {task_metric['total_tokens']:,}")
        print(f"  Cost: ${task_metric['total_cost']:.4f}")
        print(f"  Lines of code: {task_metric['total_lines_of_code']}")
        print(f"  Complexity: {task_metric['total_complexity']}")
        print(f"  Tools used: {task_metric['unique_tools_used']}")
        print(f"  Efficiency score: {task_metric['efficiency_score']:.1f}/100")
        print(f"  Quality score: {task_metric['quality_score']:.1f}/100")
        
        # Action distribution
        actions = task_metric['action_distribution']
        if actions:
            print(f"  Action distribution: {dict(list(actions.items())[:3])}")
        
        # LLM evaluation results for this task
        if hasattr(evaluator, 'llm_evaluation_results') and evaluator.llm_evaluation_results:
            task_key = f'task_{task_id}'
            if 'task_results' in evaluator.llm_evaluation_results and task_key in evaluator.llm_evaluation_results['task_results']:
                llm_task_results = evaluator.llm_evaluation_results['task_results'][task_key]
                print(f"  LLM Evaluation Scores:")
                for metric, result in llm_task_results.items():
                    print(f"    {metric.replace('_', ' ').title()}: {result['score']:.1f}/10 (confidence: {result['confidence']:.2f})")


def validate_dependencies():
    """Validate that required dependencies are available."""
    required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Error: Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)


if __name__ == '__main__':
    # Validate dependencies
    validate_dependencies()
    
    # Run main function
    main() 