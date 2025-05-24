"""
LLM-based Evaluation Module

This module provides LLM-based evaluation for abstract and difficult-to-quantify metrics
such as consistency, planning quality, tool selection efficiency, and code quality.
"""

import json
import os
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Enumeration of available evaluation metrics."""
    CONSISTENCY = "consistency"
    STEP_WISE_CORRECTNESS = "step_wise_correctness"
    TOOL_SELECTION_EFFICIENCY = "tool_selection_efficiency"
    PLANNING_QUALITY = "planning_quality"
    INFORMATION_RETRIEVAL_EFFICIENCY = "information_retrieval_efficiency"
    CODE_QUALITY = "code_quality"
    LEARNING_ADAPTABILITY = "learning_adaptability"
    ERROR_HANDLING_QUALITY = "error_handling_quality"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    COMMUNICATION_CLARITY = "communication_clarity"
    GOAL_ALIGNMENT = "goal_alignment"
    CREATIVITY_INNOVATION = "creativity_innovation"


@dataclass
class EvaluationResult:
    """Result of an LLM evaluation."""
    metric: EvaluationMetric
    score: float  # 1-10 scale
    reasoning: str
    confidence: float  # 0-1 scale
    details: Dict[str, Any] = None


@dataclass
class EvaluationContext:
    """Context information for evaluation."""
    task_id: int
    task_description: str
    steps: List[Dict[str, Any]]
    code_generated: str = ""
    tools_available: List[str] = None
    errors_encountered: List[Dict[str, Any]] = None
    similar_tasks: List[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    async def evaluate(self, prompt: str, context: EvaluationContext) -> Dict[str, Any]:
        """Evaluate using the LLM."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for evaluation."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        return self.api_key is not None
    
    async def evaluate(self, prompt: str, context: EvaluationContext) -> Dict[str, Any]:
        """Evaluate using OpenAI GPT."""
        try:
            import openai
            
            if not self.validate_config():
                raise ValueError("OpenAI API key not provided")
            
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI agent performance. Provide detailed, objective evaluations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return {"response": response.choices[0].message.content}
            
        except ImportError:
            logger.error("OpenAI library not installed. Install with: pip install openai")
            return {"error": "OpenAI library not available"}
        except Exception as e:
            logger.error(f"OpenAI evaluation error: {str(e)}")
            return {"error": str(e)}


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for evaluation."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        return self.api_key is not None
    
    async def evaluate(self, prompt: str, context: EvaluationContext) -> Dict[str, Any]:
        """Evaluate using Anthropic Claude."""
        try:
            import anthropic
            
            if not self.validate_config():
                raise ValueError("Anthropic API key not provided")
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            response = await client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {"response": response.content[0].text}
            
        except ImportError:
            logger.error("Anthropic library not installed. Install with: pip install anthropic")
            return {"error": "Anthropic library not available"}
        except Exception as e:
            logger.error(f"Anthropic evaluation error: {str(e)}")
            return {"error": str(e)}


class GeminiProvider(LLMProvider):
    """Google Gemini provider for evaluation."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    def validate_config(self) -> bool:
        """Validate Gemini configuration."""
        return self.api_key is not None
    
    async def evaluate(self, prompt: str, context: EvaluationContext) -> Dict[str, Any]:
        """Evaluate using Google Gemini."""
        try:
            import google.generativeai as genai
            
            if not self.validate_config():
                raise ValueError("Google API key not provided")
            
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Create the model
            model = genai.GenerativeModel(self.model)
            
            # Prepare the full prompt with system instruction
            full_prompt = f"""You are an expert evaluator of AI agent performance. Provide detailed, objective evaluations.

{prompt}"""
            
            # Generate response asynchronously
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                )
            )
            
            return {"response": response.text}
            
        except ImportError:
            logger.error("Google Generative AI library not installed. Install with: pip install google-generativeai")
            return {"error": "Google Generative AI library not available"}
        except Exception as e:
            logger.error(f"Gemini evaluation error: {str(e)}")
            return {"error": str(e)}


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing and development."""
    
    def __init__(self, api_key: str = None, model: str = "mock"):
        super().__init__(api_key, model)
    
    def validate_config(self) -> bool:
        """Always valid for mock provider."""
        return True
    
    async def evaluate(self, prompt: str, context: EvaluationContext) -> Dict[str, Any]:
        """Mock evaluation with realistic but fake responses."""
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Generate mock response based on the metric being evaluated
        if "consistency" in prompt.lower():
            score = 7.5
            reasoning = "The agent shows good consistency in approach but some variation in resource usage."
        elif "code quality" in prompt.lower():
            score = 8.2
            reasoning = "Code is well-structured with good readability and follows best practices."
        elif "tool selection" in prompt.lower():
            score = 6.8
            reasoning = "Tool selection is generally appropriate but could be more efficient in some cases."
        else:
            score = 7.0
            reasoning = "Performance is above average with room for improvement."
        
        return {
            "response": json.dumps({
                "score": score,
                "reasoning": reasoning,
                "confidence": 0.8,
                "details": {"mock": True}
            })
        }


class PromptTemplates:
    """Collection of prompt templates for different evaluation metrics."""
    
    @staticmethod
    def get_consistency_prompt(context: EvaluationContext) -> str:
        """Generate prompt for consistency evaluation."""
        return f"""
Evaluate the CONSISTENCY of the agent's approach across similar tasks.

Task ID: {context.task_id}
Task Description: {context.task_description}

Context:
- Number of steps taken: {len(context.steps)}
- Code generated length: {len(context.code_generated)} characters
- Tools used: {[step.get('tool_used') for step in context.steps if step.get('tool_used')]}

Similar tasks for comparison:
{json.dumps(context.similar_tasks, indent=2) if context.similar_tasks else "No similar tasks provided"}

Evaluation Criteria:
1. Approach consistency across similar problems
2. Resource usage patterns
3. Tool selection consistency
4. Solution structure similarity

Please evaluate on a scale of 1-10 where:
- 1-3: Highly inconsistent, different approaches for similar problems
- 4-6: Somewhat consistent, minor variations acceptable
- 7-9: Very consistent, similar approaches with justified variations
- 10: Perfect consistency with optimal adaptations

Return your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "confidence": <number 0-1>,
    "details": {{
        "approach_similarity": <number 1-10>,
        "resource_usage_consistency": <number 1-10>,
        "tool_selection_consistency": <number 1-10>
    }}
}}
"""
    
    @staticmethod
    def get_step_wise_correctness_prompt(context: EvaluationContext) -> str:
        """Generate prompt for step-wise correctness evaluation."""
        steps_detail = "\n".join([
            f"Step {i+1}: {step.get('action', 'unknown')} - {step.get('tool_used', 'no tool')} - Duration: {step.get('duration_seconds', 0):.2f}s"
            for i, step in enumerate(context.steps[:10])  # Limit to first 10 steps
        ])
        
        return f"""
Evaluate the STEP-WISE CORRECTNESS of the agent's decision-making process.

Task: {context.task_description}

Steps taken:
{steps_detail}

Total steps: {len(context.steps)}
Errors encountered: {len(context.errors_encountered) if context.errors_encountered else 0}

Evaluation Criteria:
1. Logical progression of steps
2. Appropriateness of each decision
3. Efficiency of the solution path
4. Proper error handling and recovery

Please evaluate each aspect and provide an overall score (1-10):
- 1-3: Poor decisions, illogical progression
- 4-6: Adequate decisions with some inefficiencies
- 7-9: Good decisions, logical and efficient
- 10: Excellent decisions, optimal progression

Return your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "confidence": <number 0-1>,
    "details": {{
        "logical_progression": <number 1-10>,
        "decision_appropriateness": <number 1-10>,
        "solution_efficiency": <number 1-10>,
        "error_handling": <number 1-10>
    }}
}}
"""
    
    @staticmethod
    def get_tool_selection_efficiency_prompt(context: EvaluationContext) -> str:
        """Generate prompt for tool selection efficiency evaluation."""
        tools_used = [step.get('tool_used') for step in context.steps if step.get('tool_used')]
        tools_used_counts = {}
        for tool in tools_used:
            tools_used_counts[tool] = tools_used_counts.get(tool, 0) + 1
        
        return f"""
Evaluate the TOOL SELECTION EFFICIENCY of the agent.

Task: {context.task_description}

Available tools: {context.tools_available or "Not specified"}
Tools actually used: {list(tools_used_counts.keys())}
Tool usage frequency: {tools_used_counts}

Evaluation Criteria:
1. Appropriateness of tool choices for each step
2. Efficiency vs. alternative tools
3. Proper utilization of available tools
4. Avoidance of redundant tool usage

Please evaluate on a scale of 1-10:
- 1-3: Poor tool choices, inefficient usage
- 4-6: Adequate tool selection with some suboptimal choices
- 7-9: Good tool selection, efficient and appropriate
- 10: Optimal tool selection, perfect efficiency

Return your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "confidence": <number 0-1>,
    "details": {{
        "tool_appropriateness": <number 1-10>,
        "efficiency_vs_alternatives": <number 1-10>,
        "utilization_completeness": <number 1-10>,
        "redundancy_avoidance": <number 1-10>
    }}
}}
"""
    
    @staticmethod
    def get_planning_quality_prompt(context: EvaluationContext) -> str:
        """Generate prompt for planning quality evaluation."""
        return f"""
Evaluate the PLANNING QUALITY and solution approach coherence.

Task: {context.task_description}

Solution approach analysis:
- Total steps: {len(context.steps)}
- Planning-related steps: {len([s for s in context.steps if 'think' in str(s.get('tool_used', '')).lower()])}
- Average step duration: {sum(s.get('duration_seconds', 0) for s in context.steps) / len(context.steps):.2f}s

Step sequence (first 10):
{chr(10).join([f"{i+1}. {step.get('action', 'unknown')}" for i, step in enumerate(context.steps[:10])])}

Evaluation Criteria:
1. Solution structure and coherence
2. Logical flow and sequencing
3. Completeness of planning
4. Adaptability to obstacles

Please evaluate on a scale of 1-10:
- 1-3: Poor planning, incoherent approach
- 4-6: Basic planning with some structure
- 7-9: Good planning, coherent and logical
- 10: Excellent planning, comprehensive and adaptive

Return your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "confidence": <number 0-1>,
    "details": {{
        "solution_structure": <number 1-10>,
        "logical_flow": <number 1-10>,
        "planning_completeness": <number 1-10>,
        "adaptability": <number 1-10>
    }}
}}
"""
    
    @staticmethod
    def get_code_quality_prompt(context: EvaluationContext) -> str:
        """Generate prompt for code quality evaluation."""
        return f"""
Evaluate the CODE QUALITY of generated code.

Task: {context.task_description}

Generated Code:
{context.code_generated[:2000]}...  # Truncated for brevity

Code Statistics:
- Length: {len(context.code_generated)} characters
- Lines: {len(context.code_generated.split(chr(10)))} lines

Evaluation Criteria:
1. Readability and clarity
2. Maintainability and structure
3. Efficiency and performance
4. Best practices adherence
5. Documentation and comments

Please evaluate on a scale of 1-10:
- 1-3: Poor code quality, hard to read/maintain
- 4-6: Adequate code quality with some issues
- 7-9: Good code quality, well-structured
- 10: Excellent code quality, exemplary practices

Return your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "confidence": <number 0-1>,
    "details": {{
        "readability": <number 1-10>,
        "maintainability": <number 1-10>,
        "efficiency": <number 1-10>,
        "best_practices": <number 1-10>,
        "documentation": <number 1-10>
    }}
}}
"""
    
    @staticmethod
    def get_learning_adaptability_prompt(context: EvaluationContext) -> str:
        """Generate prompt for learning adaptability evaluation."""
        return f"""
Evaluate the LEARNING ADAPTABILITY across similar tasks.

Current Task: {context.task_description}

Similar tasks comparison:
{json.dumps(context.similar_tasks, indent=2) if context.similar_tasks else "No comparison data available"}

Evaluation Criteria:
1. Performance improvement across similar tasks
2. Adaptation of successful strategies
3. Learning from previous errors
4. Optimization of resource usage

Please evaluate on a scale of 1-10:
- 1-3: No learning evident, repeated mistakes
- 4-6: Some learning, minor improvements
- 7-9: Good learning, clear improvements
- 10: Excellent learning, optimal adaptation

Return your evaluation in this JSON format:
{{
    "score": <number 1-10>,
    "reasoning": "<detailed explanation>",
    "confidence": <number 0-1>,
    "details": {{
        "performance_improvement": <number 1-10>,
        "strategy_adaptation": <number 1-10>,
        "error_learning": <number 1-10>,
        "resource_optimization": <number 1-10>
    }}
}}
"""


class LLMEvaluator:
    """Main LLM evaluation coordinator."""
    
    def __init__(self, provider: LLMProvider, config: Dict[str, Any] = None):
        self.provider = provider
        self.config = config or {}
        self.prompt_templates = PromptTemplates()
        
        if not self.provider.validate_config():
            logger.warning("LLM provider configuration is invalid")
    
    async def evaluate_metric(self, metric: EvaluationMetric, context: EvaluationContext) -> EvaluationResult:
        """Evaluate a specific metric using LLM."""
        try:
            # Get appropriate prompt
            prompt = self._get_prompt_for_metric(metric, context)
            
            # Call LLM
            response = await self.provider.evaluate(prompt, context)
            
            if "error" in response:
                logger.error(f"LLM evaluation error for {metric.value}: {response['error']}")
                return EvaluationResult(
                    metric=metric,
                    score=5.0,  # Default neutral score
                    reasoning=f"Evaluation failed: {response['error']}",
                    confidence=0.0
                )
            
            # Parse response
            return self._parse_llm_response(metric, response["response"])
            
        except Exception as e:
            logger.error(f"Error evaluating {metric.value}: {str(e)}")
            return EvaluationResult(
                metric=metric,
                score=5.0,
                reasoning=f"Evaluation error: {str(e)}",
                confidence=0.0
            )
    
    async def evaluate_all_metrics(self, context: EvaluationContext) -> List[EvaluationResult]:
        """Evaluate all available metrics."""
        results = []
        
        # Define which metrics to evaluate based on available data
        metrics_to_evaluate = [
            EvaluationMetric.STEP_WISE_CORRECTNESS,
            EvaluationMetric.TOOL_SELECTION_EFFICIENCY,
            EvaluationMetric.PLANNING_QUALITY,
        ]
        
        # Add code quality if code is available
        if context.code_generated:
            metrics_to_evaluate.append(EvaluationMetric.CODE_QUALITY)
        
        # Add consistency and learning adaptability if similar tasks available
        if context.similar_tasks:
            metrics_to_evaluate.extend([
                EvaluationMetric.CONSISTENCY,
                EvaluationMetric.LEARNING_ADAPTABILITY
            ])
        
        # Evaluate each metric
        for metric in metrics_to_evaluate:
            result = await self.evaluate_metric(metric, context)
            results.append(result)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        return results
    
    def _get_prompt_for_metric(self, metric: EvaluationMetric, context: EvaluationContext) -> str:
        """Get the appropriate prompt for a metric."""
        prompt_map = {
            EvaluationMetric.CONSISTENCY: self.prompt_templates.get_consistency_prompt,
            EvaluationMetric.STEP_WISE_CORRECTNESS: self.prompt_templates.get_step_wise_correctness_prompt,
            EvaluationMetric.TOOL_SELECTION_EFFICIENCY: self.prompt_templates.get_tool_selection_efficiency_prompt,
            EvaluationMetric.PLANNING_QUALITY: self.prompt_templates.get_planning_quality_prompt,
            EvaluationMetric.CODE_QUALITY: self.prompt_templates.get_code_quality_prompt,
            EvaluationMetric.LEARNING_ADAPTABILITY: self.prompt_templates.get_learning_adaptability_prompt,
        }
        
        if metric in prompt_map:
            return prompt_map[metric](context)
        else:
            return f"Evaluate the {metric.value} for the given task context."
    
    def _parse_llm_response(self, metric: EvaluationMetric, response: str) -> EvaluationResult:
        """Parse LLM response into structured result."""
        try:
            # Try to parse JSON response
            if response.strip().startswith('{'):
                data = json.loads(response.strip())
                return EvaluationResult(
                    metric=metric,
                    score=float(data.get('score', 5.0)),
                    reasoning=data.get('reasoning', 'No reasoning provided'),
                    confidence=float(data.get('confidence', 0.5)),
                    details=data.get('details', {})
                )
            else:
                # Fallback: extract score from text response
                score = self._extract_score_from_text(response)
                return EvaluationResult(
                    metric=metric,
                    score=score,
                    reasoning=response[:500] + "..." if len(response) > 500 else response,
                    confidence=0.5
                )
                
        except Exception as e:
            logger.error(f"Error parsing LLM response for {metric.value}: {str(e)}")
            return EvaluationResult(
                metric=metric,
                score=5.0,
                reasoning=f"Response parsing failed: {str(e)}",
                confidence=0.0
            )
    
    def _extract_score_from_text(self, text: str) -> float:
        """Extract numerical score from text response."""
        import re
        
        # Look for patterns like "score: 7.5" or "rating: 8/10"
        patterns = [
            r'score[:\s]*([0-9]+\.?[0-9]*)',
            r'rating[:\s]*([0-9]+\.?[0-9]*)',
            r'([0-9]+\.?[0-9]*)\s*/\s*10',
            r'([0-9]+\.?[0-9]*)\s*out\s*of\s*10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return min(10.0, max(1.0, score))  # Clamp to 1-10 range
                except ValueError:
                    continue
        
        return 5.0  # Default neutral score


def create_llm_evaluator(provider_type: str = "mock", **kwargs) -> LLMEvaluator:
    """Factory function to create LLM evaluator with specified provider."""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "mock": MockLLMProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unsupported provider: {provider_type}. Available: {list(providers.keys())}")
    
    provider = providers[provider_type](**kwargs)
    return LLMEvaluator(provider) 