#!/usr/bin/env python3
"""
Comprehensive Test Suite for Step-wise Agent Evaluation Framework

This file combines all tests for both the standard evaluation framework
and the LLM-based evaluation features into one comprehensive test suite.
"""

import os
import sys
import json
import unittest
import asyncio
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from step_wise_evaluator import StepWiseEvaluator
from llm_evaluator import create_llm_evaluator, EvaluationMetric, EvaluationContext


class TestStepWiseEvaluator(unittest.TestCase):
    """Test cases for the core StepWiseEvaluator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.test_data_path = os.path.join(os.path.dirname(__file__), 'data', 'testv1.json')
        
        # Create a small sample data for unit tests
        cls.sample_data = [
            {
                "id": 3,
                "timestamp": "2025-05-24T18:51:57.281362",
                "source": "environment",
                "message": "",
                "observation": "agent_state_changed"
            },
            {
                "id": 4,
                "timestamp": "2025-05-24T18:52:01.985679",
                "source": "user",
                "message": "Create a dice rolling program",
                "action": "message"
            },
            {
                "id": 5,
                "timestamp": "2025-05-24T18:52:02.303455",
                "source": "agent",
                "action": "edit",
                "args": {
                    "file_text": "import random\n\ndef roll_dice():\n    return random.randint(1, 6)",
                    "path": "/workspace/dice.py"
                },
                "llm_metrics": {
                    "accumulated_cost": 0.01,
                    "accumulated_token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "per_turn_token": 150
                    }
                }
            },
            {
                "id": 6,
                "timestamp": "2025-05-24T18:52:10.000000",
                "source": "agent",
                "action": "finish",
                "args": {
                    "task_completed": "true",
                    "message": "Task completed successfully"
                }
            }
        ]
    
    def test_initialization_with_data(self):
        """Test initialization with direct data."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        self.assertIsNotNone(evaluator.data)
        self.assertEqual(len(evaluator.data), 4)
    
    def test_initialization_with_file(self):
        """Test initialization with file path."""
        if os.path.exists(self.test_data_path):
            evaluator = StepWiseEvaluator(data_path=self.test_data_path)
            self.assertIsNotNone(evaluator.data)
            self.assertGreater(len(evaluator.data), 0)
    
    def test_initialization_with_llm_config(self):
        """Test initialization with LLM configuration."""
        llm_config = {'provider': 'mock'}
        evaluator = StepWiseEvaluator(data=self.sample_data, llm_config=llm_config)
        self.assertIsNotNone(evaluator.llm_evaluator)
    
    def test_task_boundary_identification(self):
        """Test task boundary identification."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        self.assertIn(1, evaluator.task_boundaries)
        self.assertEqual(evaluator.task_boundaries[1]['start_id'], 4)
    
    def test_step_metrics_extraction(self):
        """Test step metrics extraction."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        self.assertGreater(len(evaluator.step_metrics), 0)
        
        # Check if metrics contain expected fields
        step = evaluator.step_metrics[0]
        expected_fields = ['step_id', 'task_id', 'timestamp', 'source', 'action', 
                          'duration_seconds', 'is_error', 'tool_used', 'tokens_total']
        for field in expected_fields:
            self.assertIn(field, step)
    
    def test_error_detection(self):
        """Test error detection in steps."""
        error_data = [
            {
                "id": 3,
                "timestamp": "2025-05-24T18:51:57.281362",
                "source": "agent",
                "message": "Error: Command failed",
                "action": "run"
            }
        ]
        evaluator = StepWiseEvaluator(data=error_data)
        self.assertTrue(evaluator._is_error_step(error_data[0]))
    
    def test_lines_of_code_counting(self):
        """Test lines of code counting."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        entry_with_code = {
            "args": {
                "file_text": "line1\nline2\nline3"
            }
        }
        lines = evaluator._count_lines_of_code(entry_with_code)
        self.assertEqual(lines, 3)
    
    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        entry_with_code = {
            "args": {
                "file_text": "def func():\n    if True:\n        for i in range(10):\n            pass"
            }
        }
        complexity = evaluator._calculate_complexity_score(entry_with_code)
        self.assertGreater(complexity, 0)
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        metrics = evaluator.get_efficiency_metrics()
        
        expected_keys = ['average_steps_per_task', 'average_time_per_task', 
                        'average_time_per_step', 'total_steps_all_tasks']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_error_recovery_metrics(self):
        """Test error recovery metrics calculation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        metrics = evaluator.get_error_recovery_metrics()
        
        expected_keys = ['total_errors', 'average_errors_per_task', 
                        'error_recovery_rate', 'average_error_rate']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_task_completion_metrics(self):
        """Test task completion metrics calculation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        metrics = evaluator.get_task_completion_metrics()
        
        expected_keys = ['total_tasks', 'completed_tasks', 'completion_rate', 
                        'total_tokens_used', 'total_cost']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_code_quality_metrics(self):
        """Test code quality metrics calculation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        metrics = evaluator.get_code_quality_metrics()
        
        expected_keys = ['total_lines_of_code', 'average_lines_per_task', 
                        'total_complexity', 'average_quality_score']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_tool_usage_metrics(self):
        """Test tool usage metrics calculation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        metrics = evaluator.get_tool_usage_metrics()
        
        expected_keys = ['total_tool_calls', 'unique_tools_used', 
                        'tool_distribution', 'most_used_tools']
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        report = evaluator.generate_comprehensive_report()
        
        expected_sections = ['task_boundaries', 'efficiency_metrics', 
                           'error_recovery_metrics', 'task_completion_metrics',
                           'code_quality_metrics', 'tool_usage_metrics']
        for section in expected_sections:
            self.assertIn(section, report)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualization_creation(self, mock_close, mock_savefig):
        """Test visualization creation (mocked)."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        
        # Mock the visualization creation
        evaluator.create_visualizations('test_output')
        
        # Check if savefig was called (indicating plots were created)
        self.assertTrue(mock_savefig.called)
    
    def test_report_saving(self):
        """Test report saving to file."""
        evaluator = StepWiseEvaluator(data=self.sample_data)
        
        # Create a temporary file path
        temp_file = 'test_report.json'
        
        try:
            evaluator.save_report(temp_file)
            
            # Check if file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Check if file contains valid JSON
            with open(temp_file, 'r') as f:
                report_data = json.load(f)
                self.assertIsInstance(report_data, dict)
        
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestLLMEvaluator(unittest.TestCase):
    """Test cases for LLM evaluation functionality."""
    
    def setUp(self):
        """Set up for LLM tests."""
        self.sample_context = EvaluationContext(
            task_id=1,
            task_description="Create a dice rolling program",
            steps=[
                {'action': 'edit', 'tool_used': 'str_replace_editor', 'duration_seconds': 5.0},
                {'action': 'run', 'tool_used': 'execute_bash', 'duration_seconds': 2.0},
                {'action': 'finish', 'tool_used': None, 'duration_seconds': 1.0}
            ],
            code_generated="import random\n\ndef roll_dice():\n    return random.randint(1, 6)",
            tools_available=['str_replace_editor', 'execute_bash', 'think'],
            errors_encountered=[],
            similar_tasks=[]
        )
    
    def test_mock_llm_provider_creation(self):
        """Test creation of mock LLM provider."""
        llm_evaluator = create_llm_evaluator('mock')
        self.assertIsNotNone(llm_evaluator)
        self.assertTrue(llm_evaluator.provider.validate_config())
    
    def test_openai_provider_creation(self):
        """Test creation of OpenAI provider (without API key)."""
        try:
            llm_evaluator = create_llm_evaluator('openai', api_key='fake-key')
            self.assertIsNotNone(llm_evaluator)
        except Exception as e:
            # Expected to fail without real API key
            self.assertIn("openai", str(e).lower())
    
    def test_anthropic_provider_creation(self):
        """Test creation of Anthropic provider (without API key)."""
        try:
            llm_evaluator = create_llm_evaluator('anthropic', api_key='fake-key')
            self.assertIsNotNone(llm_evaluator)
        except Exception as e:
            # Expected to fail without real API key
            self.assertIn("anthropic", str(e).lower())
    
    def test_gemini_provider_creation(self):
        """Test creation of Gemini provider (without API key)."""
        try:
            llm_evaluator = create_llm_evaluator('gemini', api_key='fake-key')
            self.assertIsNotNone(llm_evaluator)
        except Exception as e:
            # Expected to fail without real API key
            self.assertIn("gemini", str(e).lower())
    
    def test_invalid_provider_creation(self):
        """Test creation with invalid provider."""
        with self.assertRaises(ValueError):
            create_llm_evaluator('invalid_provider')
    
    def test_evaluation_context_creation(self):
        """Test evaluation context creation."""
        context = self.sample_context
        self.assertEqual(context.task_id, 1)
        self.assertEqual(len(context.steps), 3)
        self.assertIsNotNone(context.code_generated)


class TestLLMEvaluatorAsync(unittest.IsolatedAsyncioTestCase):
    """Async test cases for LLM evaluation functionality."""
    
    async def test_mock_llm_evaluation(self):
        """Test LLM evaluation with mock provider."""
        llm_evaluator = create_llm_evaluator('mock')
        
        context = EvaluationContext(
            task_id=1,
            task_description="Create a dice rolling program",
            steps=[
                {'action': 'edit', 'tool_used': 'str_replace_editor', 'duration_seconds': 5.0},
                {'action': 'run', 'tool_used': 'execute_bash', 'duration_seconds': 2.0}
            ],
            code_generated="import random\n\ndef roll_dice():\n    return random.randint(1, 6)",
            tools_available=['str_replace_editor', 'execute_bash', 'think'],
            errors_encountered=[],
            similar_tasks=[]
        )
        
        # Test individual metric evaluation
        result = await llm_evaluator.evaluate_metric(EvaluationMetric.STEP_WISE_CORRECTNESS, context)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.metric, EvaluationMetric.STEP_WISE_CORRECTNESS)
        self.assertGreaterEqual(result.score, 1.0)
        self.assertLessEqual(result.score, 10.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    async def test_multiple_metrics_evaluation(self):
        """Test evaluation of multiple metrics."""
        llm_evaluator = create_llm_evaluator('mock')
        
        context = EvaluationContext(
            task_id=1,
            task_description="Create a dice rolling program",
            steps=[
                {'action': 'edit', 'tool_used': 'str_replace_editor', 'duration_seconds': 5.0}
            ],
            code_generated="import random\n\ndef roll_dice():\n    return random.randint(1, 6)",
            tools_available=['str_replace_editor', 'execute_bash'],
            errors_encountered=[],
            similar_tasks=[]
        )
        
        # Test multiple metrics
        metrics_to_test = [
            EvaluationMetric.STEP_WISE_CORRECTNESS,
            EvaluationMetric.TOOL_SELECTION_EFFICIENCY,
            EvaluationMetric.PLANNING_QUALITY,
            EvaluationMetric.CODE_QUALITY
        ]
        
        results = []
        for metric in metrics_to_test:
            result = await llm_evaluator.evaluate_metric(metric, context)
            results.append(result)
        
        self.assertEqual(len(results), len(metrics_to_test))
        for result in results:
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.score, 1.0)
            self.assertLessEqual(result.score, 10.0)


class TestRealDataEvaluation(unittest.TestCase):
    """Test the evaluator with real data if available."""
    
    def setUp(self):
        """Set up for real data tests."""
        self.test_data_path = os.path.join(os.path.dirname(__file__), 'data', 'testv1.json')
    
    def test_real_data_processing(self):
        """Test processing of real JSON data."""
        if not os.path.exists(self.test_data_path):
            self.skipTest("Real data file not found")
        
        evaluator = StepWiseEvaluator(data_path=self.test_data_path)
        
        # Basic checks
        self.assertGreater(len(evaluator.data), 0)
        self.assertGreater(len(evaluator.task_boundaries), 0)
        self.assertGreater(len(evaluator.step_metrics), 0)
        
        # Check that we have 6 tasks as expected
        self.assertEqual(len(evaluator.task_boundaries), 6)
        
        # Generate report
        report = evaluator.generate_comprehensive_report()
        self.assertIsInstance(report, dict)
        
        # Print some basic statistics
        print(f"\nReal Data Statistics:")
        print(f"Total events processed: {len(evaluator.data)}")
        print(f"Total tasks identified: {len(evaluator.task_boundaries)}")
        print(f"Total steps analyzed: {len(evaluator.step_metrics)}")
        
        # Check efficiency metrics
        eff_metrics = report['efficiency_metrics']
        print(f"Average steps per task: {eff_metrics['average_steps_per_task']:.1f}")
        print(f"Average time per task: {eff_metrics['average_time_per_task']:.1f} seconds")
        
        # Check completion metrics
        comp_metrics = report['task_completion_metrics']
        print(f"Task completion rate: {comp_metrics['completion_rate']:.1f}%")
        print(f"Total tokens used: {comp_metrics['total_tokens_used']:,}")


class TestIntegrationLLMFramework(unittest.IsolatedAsyncioTestCase):
    """Integration tests for LLM evaluation with the main framework."""
    
    async def test_integration_with_existing_framework(self):
        """Test integration with existing evaluation framework."""
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'testv1.json')
        
        if not os.path.exists(data_path):
            self.skipTest("Test data file not found")
        
        # Test without LLM
        evaluator_no_llm = StepWiseEvaluator(data_path=data_path)
        report_no_llm = evaluator_no_llm.generate_comprehensive_report()
        
        self.assertIsInstance(report_no_llm, dict)
        self.assertNotIn('llm_evaluation_metrics', report_no_llm)
        
        # Test with LLM (mock)
        llm_config = {'provider': 'mock'}
        evaluator_with_llm = StepWiseEvaluator(data_path=data_path, llm_config=llm_config)
        
        # Run async LLM evaluation
        llm_results = await evaluator_with_llm.run_llm_evaluation()
        evaluator_with_llm.llm_evaluation_results = llm_results
        
        report_with_llm = evaluator_with_llm.generate_comprehensive_report(include_llm_evaluation=True)
        
        self.assertIsInstance(report_with_llm, dict)
        self.assertIn('llm_evaluation_metrics', report_with_llm)
        
        # Verify LLM results structure
        llm_metrics = report_with_llm['llm_evaluation_metrics']
        self.assertIn('evaluation_summary', llm_metrics)
        self.assertIn('aggregate_scores', llm_metrics)
        self.assertIn('task_results', llm_metrics)
    
    async def test_llm_evaluation_with_specific_metrics(self):
        """Test LLM evaluation with specific metrics selection."""
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'testv1.json')
        
        if not os.path.exists(data_path):
            self.skipTest("Test data file not found")
        
        llm_config = {'provider': 'mock'}
        evaluator = StepWiseEvaluator(data_path=data_path, llm_config=llm_config)
        
        # Test specific metrics
        selected_metrics = ['step_wise_correctness', 'planning_quality']
        results = await evaluator.run_llm_evaluation(selected_metrics)
        
        self.assertNotIn('error', results)
        self.assertIn('evaluation_summary', results)
        self.assertEqual(
            set(results['evaluation_summary']['metrics_evaluated']),
            set(selected_metrics)
        )


def run_full_evaluation():
    """Run a full evaluation on the real data and generate outputs."""
    test_data_path = os.path.join(os.path.dirname(__file__), 'data', 'testv1.json')
    
    if not os.path.exists(test_data_path):
        print("Error: Test data file not found at", test_data_path)
        return
    
    print("Running full evaluation on real data...")
    
    # Create evaluator
    evaluator = StepWiseEvaluator(data_path=test_data_path)
    
    # Generate and save report
    report_path = 'evaluation_report.json'
    evaluator.save_report(report_path)
    
    # Create visualizations
    viz_dir = 'evaluation_visualizations'
    evaluator.create_visualizations(viz_dir)
    
    # Print summary
    evaluator.print_summary()
    
    print(f"\nEvaluation complete!")
    print(f"Report saved to: {report_path}")
    print(f"Visualizations saved to: {viz_dir}/")


async def run_llm_evaluation_demo():
    """Run LLM evaluation demonstration."""
    test_data_path = os.path.join(os.path.dirname(__file__), 'data', 'testv1.json')
    
    if not os.path.exists(test_data_path):
        print("Error: Test data file not found at", test_data_path)
        return
    
    print("Running LLM evaluation demonstration...")
    
    # Configure mock LLM
    llm_config = {'provider': 'mock'}
    
    # Create evaluator with LLM configuration
    evaluator = StepWiseEvaluator(data_path=test_data_path, llm_config=llm_config)
    
    print(f"✓ Loaded {len(evaluator.data)} events")
    print(f"✓ Identified {len(evaluator.task_boundaries)} tasks")
    print(f"✓ LLM evaluator initialized: {evaluator.llm_evaluator is not None}")
    
    # Run LLM evaluation
    print("\nRunning LLM evaluation...")
    llm_results = await evaluator.run_llm_evaluation()
    
    if 'error' in llm_results:
        print(f"✗ LLM evaluation failed: {llm_results['error']}")
        return
    
    print(f"✓ LLM evaluation completed")
    print(f"✓ Tasks evaluated: {llm_results['evaluation_summary']['total_tasks_evaluated']}")
    print(f"✓ Metrics evaluated: {', '.join(llm_results['evaluation_summary']['metrics_evaluated'])}")
    
    # Display results
    print("\nAGGREGATE SCORES:")
    for metric, scores in llm_results['aggregate_scores'].items():
        print(f"  {metric.replace('_', ' ').title()}: {scores['average_score']:.2f}/10")
    
    # Save enhanced report
    evaluator.llm_evaluation_results = llm_results
    enhanced_report = evaluator.generate_comprehensive_report(include_llm_evaluation=True)
    
    os.makedirs('test_output', exist_ok=True)
    with open('test_output/enhanced_evaluation_report.json', 'w') as f:
        json.dump(enhanced_report, f, indent=2, default=str)
    
    print(f"\n✓ Enhanced report saved to test_output/enhanced_evaluation_report.json")


def run_configuration_tests():
    """Test different LLM provider configurations."""
    print("Testing LLM provider configurations...")
    
    configurations = [
        {'provider': 'mock'},
        {'provider': 'mock', 'model': 'mock-advanced'},
        {'provider': 'openai', 'api_key': 'fake-key'},  # Will fail gracefully
        {'provider': 'anthropic', 'api_key': 'fake-key'},  # Will fail gracefully
        {'provider': 'gemini', 'api_key': 'fake-key'},  # Will fail gracefully
    ]
    
    for i, config in enumerate(configurations, 1):
        print(f"\nTesting configuration {i}: {config}")
        try:
            llm_evaluator = create_llm_evaluator(**config)
            print(f"  ✓ LLM evaluator created: {llm_evaluator.provider.__class__.__name__}")
            print(f"  ✓ Configuration valid: {llm_evaluator.provider.validate_config()}")
        except Exception as e:
            print(f"  ✗ Configuration failed: {str(e)}")


async def main():
    """Run comprehensive tests and demonstrations."""
    print("COMPREHENSIVE STEP-WISE EVALUATION FRAMEWORK TESTS")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run unit tests
    print("\n" + "="*20 + " UNIT TESTS " + "="*20)
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run configuration tests
    print("\n" + "="*20 + " CONFIGURATION TESTS " + "="*20)
    run_configuration_tests()
    
    # Run LLM evaluation demo
    print("\n" + "="*20 + " LLM EVALUATION DEMO " + "="*20)
    await run_llm_evaluation_demo()
    
    # Run full evaluation
    print("\n" + "="*20 + " FULL EVALUATION " + "="*20)
    run_full_evaluation()
    
    print("\n" + "="*70)
    print("🎉 All tests and demonstrations completed!")
    print("Check the test_output/ directory for generated reports and visualizations.")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Test Suite for Step-wise Evaluation Framework')
    parser.add_argument('--unit-tests', action='store_true', 
                       help='Run unit tests only')
    parser.add_argument('--llm-tests', action='store_true', 
                       help='Run LLM evaluation tests only')
    parser.add_argument('--full-eval', action='store_true', 
                       help='Run full evaluation on real data')
    parser.add_argument('--config-tests', action='store_true', 
                       help='Run configuration tests only')
    
    args = parser.parse_args()
    
    if args.unit_tests:
        unittest.main(argv=[''], verbosity=2)
    elif args.llm_tests:
        asyncio.run(run_llm_evaluation_demo())
    elif args.full_eval:
        run_full_evaluation()
    elif args.config_tests:
        run_configuration_tests()
    else:
        # Run comprehensive test suite
        asyncio.run(main()) 