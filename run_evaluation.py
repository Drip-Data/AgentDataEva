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

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from step_wise_evaluator import StepWiseEvaluator


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
        choices=['openai', 'anthropic', 'gemini', 'mock'],
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
    
    if not args.quiet:
        print("=" * 70)
        print("STEP-WISE AGENT EVALUATION FRAMEWORK")
        print("=" * 70)
        print(f"Input file: {args.input}")
        print(f"Output directory: {args.output_dir}")
        print(f"LLM evaluation: {'Enabled' if args.llm_eval else 'Disabled'}")
        if args.llm_eval:
            print(f"LLM provider: {args.llm_provider}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
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
            print("Loading and processing data...")
        
        # Create evaluator
        evaluator = StepWiseEvaluator(data_path=args.input, llm_config=llm_config)
        
        if not args.quiet:
            print(f"✓ Loaded {len(evaluator.data)} events")
            print(f"✓ Identified {len(evaluator.task_boundaries)} tasks")
            print(f"✓ Extracted {len(evaluator.step_metrics)} step metrics")
            print()
        
        # Run LLM evaluation if enabled
        if args.llm_eval and evaluator.llm_evaluator:
            if not args.quiet:
                print("Running LLM-based evaluation...")
            
            # Run async LLM evaluation
            llm_results = asyncio.run(evaluator.run_llm_evaluation(args.llm_metrics))
            evaluator.llm_evaluation_results = llm_results
            
            if not args.quiet:
                if 'error' in llm_results:
                    print(f"⚠ LLM evaluation error: {llm_results['error']}")
                else:
                    print(f"✓ LLM evaluation completed for {llm_results['evaluation_summary']['total_tasks_evaluated']} tasks")
                    print(f"✓ Evaluated metrics: {', '.join(llm_results['evaluation_summary']['metrics_evaluated'])}")
                print()
        
        # Generate comprehensive report
        if not args.quiet:
            print("Generating comprehensive report...")
        
        report_path = os.path.join(args.output_dir, 'evaluation_report.json')
        evaluator.save_report(report_path)
        
        if not args.quiet:
            print(f"✓ Report saved to: {report_path}")
        
        # Generate visualizations
        if not args.no_viz:
            if not args.quiet:
                print("Creating visualizations...")
            
            viz_dir = os.path.join(args.output_dir, 'visualizations')
            evaluator.create_visualizations(viz_dir)
            
            if not args.quiet:
                print(f"✓ Visualizations saved to: {viz_dir}/")
        
        # Print summary
        if not args.quiet:
            print("\n" + "=" * 70)
            print("EVALUATION SUMMARY")
            print("=" * 70)
        
        evaluator.print_summary()
        
        # Print LLM evaluation summary if available
        if args.llm_eval and evaluator.llm_evaluation_results and 'aggregate_scores' in evaluator.llm_evaluation_results:
            print_llm_evaluation_summary(evaluator.llm_evaluation_results)
        
        # Print detailed task metrics if requested
        if args.detailed_tasks:
            print_detailed_task_metrics(evaluator)
        
        # Print performance statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        if not args.quiet:
            print("\n" + "=" * 70)
            print("PROCESSING STATISTICS")
            print("=" * 70)
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Events per second: {len(evaluator.data) / processing_time:.1f}")
            print(f"Output files generated:")
            print(f"  - {report_path}")
            if not args.no_viz:
                print(f"  - {viz_dir}/task_overview.png")
                print(f"  - {viz_dir}/efficiency_analysis.png")
                print(f"  - {viz_dir}/tool_usage.png")
                print(f"  - {viz_dir}/llm_usage.png")
        
        print(f"\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
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