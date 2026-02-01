#!/usr/bin/env python3
"""
Multi-Agent Orchestration with Real LLM Integration
Planning → Backend → Frontend → Testing → Deployment

This module implements a real multi-agent system where each agent uses
an LLM to perform its specialized task. Agents communicate and coordinate
to build complete software applications from requirements.

Usage:
    python src/20_multi_agent_orchestration.py --model-path /path/to/model --requirement "Build a todo app"
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_env():
    """Verify environment dependencies."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        print("[ERROR] Must be run in 'nexus' conda environment.")
        return False
    return True


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    success: bool
    data: Any
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class OrchestrationContext:
    """Shared context across all agents in the workflow."""
    requirement: str
    plan: Optional[Dict] = None
    backend_code: Optional[str] = None
    frontend_code: Optional[str] = None
    tests: Optional[str] = None
    deployment_config: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class LLMWrapper:
    """
    Wrapper for LLM inference using OmniModelLoader.
    Handles model loading, generation, and error recovery.
    """
    
    def __init__(self, model_path: str, device: str = "auto", load_in_8bit: bool = False):
        self.model_path = model_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the LLM using OmniModelLoader."""
        if self._initialized:
            return True
            
        try:
            logger.info(f"🔄 Loading LLM from {self.model_path}...")
            from src.omni.loader import OmniModelLoader
            
            loader = OmniModelLoader(self.model_path)
            self.model, self.tokenizer = loader.load(
                mode="thinker_only",
                device_map=self.device,
                load_in_8bit=self.load_in_8bit,
                trust_remote_code=True
            )
            
            self._initialized = True
            logger.info(f"✅ LLM loaded successfully: {type(self.model).__name__}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load LLM: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
        retries: int = 3
    ) -> Tuple[bool, str]:
        """
        Generate text from the LLM with retry logic.
        
        Returns:
            Tuple of (success, generated_text)
        """
        if not self._initialized and not self.initialize():
            return False, "Failed to initialize LLM"
        
        for attempt in range(retries):
            try:
                import torch
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Apply stop sequences
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text[:generated_text.index(stop_seq)]
                
                return True, generated_text.strip()
                
            except Exception as e:
                logger.warning(f"⚠️ Generation attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False, f"Generation failed after {retries} attempts: {e}"
        
        return False, "Unknown error"


class BaseAgent(ABC):
    """Base class for all agents with LLM integration."""
    
    def __init__(self, name: str, llm: LLMWrapper):
        self.name = name
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def execute(self, context: OrchestrationContext) -> AgentResult:
        """Execute the agent's task."""
        pass
    
    def _generate_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Tuple[bool, str]:
        """Generate using a structured prompt format."""
        full_prompt = f"""{system_prompt}

{user_prompt}

Response:"""
        
        return self.llm.generate(
            prompt=full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=["\n\nHuman:", "\n\nUser:", "</s>"]
        )


class PlanningAgent(BaseAgent):
    """
    Planning Agent: Breaks down requirements into actionable implementation plans.
    Uses LLM to analyze requirements and create structured development plans.
    """
    
    SYSTEM_PROMPT = """You are a Software Architecture Planning Agent. Your task is to analyze software requirements and create detailed implementation plans.

Your output must be valid JSON with the following structure:
{
    "project_name": "string",
    "description": "string",
    "components": ["list of required components"],
    "architecture": "architecture pattern (e.g., Microservices, Monolithic, Serverless)",
    "technologies": {
        "backend": ["list of backend technologies"],
        "frontend": ["list of frontend technologies"],
        "database": ["database technologies"],
        "deployment": ["deployment tools/platforms"]
    },
    "api_endpoints": [
        {"method": "GET/POST/etc", "path": "/api/...", "description": "what it does"}
    ],
    "data_models": [
        {"name": "ModelName", "fields": ["field1: type", "field2: type"]}
    ],
    "ui_components": ["list of UI components needed"],
    "timeline": "estimated timeline",
    "milestones": ["key development milestones"]
}

Be thorough and specific. Include all necessary components for a complete implementation."""

    def execute(self, context: OrchestrationContext) -> AgentResult:
        """Generate implementation plan from requirements."""
        start_time = time.time()
        self.logger.info(f"🎯 {self.name}: Analyzing requirements")
        
        user_prompt = f"""Analyze the following software requirement and create a detailed implementation plan:

Requirement: {context.requirement}

Provide your analysis as a structured JSON object following the format specified above."""
        
        success, response = self._generate_with_prompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=4096,
            temperature=0.5  # Lower temperature for structured output
        )
        
        if not success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=response,
                execution_time=time.time() - start_time
            )
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                plan_data = json.loads(response)
            
            self.logger.info(f"✅ {self.name}: Plan generated with {len(plan_data.get('components', []))} components")
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=plan_data,
                metadata={
                    "components_count": len(plan_data.get("components", [])),
                    "architecture": plan_data.get("architecture", "unknown")
                },
                execution_time=time.time() - start_time
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Failed to parse plan JSON: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=response,
                error=f"JSON parse error: {e}",
                execution_time=time.time() - start_time
            )


class BackendAgent(BaseAgent):
    """
    Backend Agent: Generates real backend code based on the implementation plan.
    Uses LLM to create complete, production-ready backend implementations.
    """
    
    SYSTEM_PROMPT = """You are a Backend Development Agent. Your task is to generate complete, production-ready backend code.

Requirements:
- Write clean, well-documented Python code using FastAPI or Flask
- Include all necessary imports
- Implement proper error handling and validation
- Use type hints where appropriate
- Include docstrings for all functions
- Follow RESTful API design principles
- Include database models if needed (using SQLAlchemy)
- Add authentication/authorization where appropriate

Your response should be complete, runnable Python code without placeholders or TODOs."""

    def execute(self, context: OrchestrationContext) -> AgentResult:
        """Generate backend code from plan."""
        start_time = time.time()
        self.logger.info(f"⚙️  {self.name}: Generating backend code")
        
        plan = context.plan or {}
        api_endpoints = json.dumps(plan.get("api_endpoints", []), indent=2)
        data_models = json.dumps(plan.get("data_models", []), indent=2)
        technologies = plan.get("technologies", {})
        backend_tech = technologies.get("backend", ["FastAPI"])
        
        user_prompt = f"""Generate complete backend code for the following project:

Project: {plan.get('project_name', 'Untitled')}
Description: {plan.get('description', context.requirement)}
Architecture: {plan.get('architecture', 'REST API')}
Backend Technologies: {', '.join(backend_tech)}

Required API Endpoints:
{api_endpoints}

Data Models:
{data_models}

Generate:
1. Main application file with all routes
2. Database models
3. Pydantic schemas for request/response validation
4. Authentication middleware (if needed)
5. Configuration management

Provide the complete, production-ready code."""
        
        success, response = self._generate_with_prompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=8192,
            temperature=0.3  # Lower temperature for code
        )
        
        if not success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=response,
                execution_time=time.time() - start_time
            )
        
        # Extract code blocks if present
        code = self._extract_code(response)
        
        self.logger.info(f"✅ {self.name}: Generated {len(code)} characters of backend code")
        
        return AgentResult(
            agent_name=self.name,
            success=True,
            data=code,
            metadata={
                "code_length": len(code),
                "language": "python"
            },
            execution_time=time.time() - start_time
        )
    
    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks or return raw response."""
        # Try to find Python code blocks
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return '\n\n'.join(matches)
        
        # Try any code blocks
        code_pattern = r'```\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return '\n\n'.join(matches)
        
        return response.strip()


class FrontendAgent(BaseAgent):
    """
    Frontend Agent: Generates real frontend code based on the implementation plan.
    Uses LLM to create complete, modern React/Vue frontend implementations.
    """
    
    SYSTEM_PROMPT = """You are a Frontend Development Agent. Your task is to generate complete, modern frontend code.

Requirements:
- Use React with TypeScript (preferred) or modern JavaScript
- Use functional components with hooks
- Include proper state management (useState, useEffect, useContext)
- Implement responsive design with CSS/Styled-components/Tailwind
- Add error boundaries and loading states
- Include prop types or TypeScript interfaces
- Write clean, maintainable code with proper component structure
- Include all necessary imports

Your response should be complete, runnable code without placeholders."""

    def execute(self, context: OrchestrationContext) -> AgentResult:
        """Generate frontend code from plan."""
        start_time = time.time()
        self.logger.info(f"🎨 {self.name}: Generating frontend code")
        
        plan = context.plan or {}
        ui_components = plan.get("ui_components", [])
        technologies = plan.get("technologies", {})
        frontend_tech = technologies.get("frontend", ["React"])
        api_endpoints = plan.get("api_endpoints", [])
        
        user_prompt = f"""Generate complete frontend code for the following project:

Project: {plan.get('project_name', 'Untitled')}
Description: {plan.get('description', context.requirement)}
Frontend Technologies: {', '.join(frontend_tech)}

Required UI Components:
{chr(10).join(f"- {comp}" for comp in ui_components)}

API Endpoints to integrate with:
{json.dumps(api_endpoints, indent=2)}

Generate:
1. Main App component
2. All necessary child components
3. API service/module for backend communication
4. State management (Context or hooks)
5. Styling (CSS, styled-components, or Tailwind)
6. Form handling and validation
7. Error handling and loading states

Provide the complete, production-ready React code with TypeScript."""
        
        success, response = self._generate_with_prompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=8192,
            temperature=0.3
        )
        
        if not success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=response,
                execution_time=time.time() - start_time
            )
        
        code = self._extract_code(response)
        
        self.logger.info(f"✅ {self.name}: Generated {len(code)} characters of frontend code")
        
        return AgentResult(
            agent_name=self.name,
            success=True,
            data=code,
            metadata={
                "code_length": len(code),
                "language": "typescript/javascript"
            },
            execution_time=time.time() - start_time
        )
    
    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks or return raw response."""
        # Try TypeScript/React first
        for lang in ['tsx', 'typescript', 'jsx', 'javascript']:
            code_pattern = rf'```{lang}\n(.*?)```'
            matches = re.findall(code_pattern, response, re.DOTALL)
            if matches:
                return '\n\n'.join(matches)
        
        # Try any code blocks
        code_pattern = r'```\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return '\n\n'.join(matches)
        
        return response.strip()


class TestingAgent(BaseAgent):
    """
    Testing Agent: Generates comprehensive tests for the generated code.
    Uses LLM to create unit tests, integration tests, and API tests.
    """
    
    SYSTEM_PROMPT = """You are a Testing Agent. Your task is to generate comprehensive test suites.

Requirements:
- Write tests using pytest for Python or Jest for JavaScript
- Include unit tests for individual functions/components
- Include integration tests for API endpoints
- Test both success and error cases
- Use mocking where appropriate for external dependencies
- Include setup and teardown fixtures
- Aim for high code coverage
- Add descriptive test names and docstrings

Your response should be complete, runnable test code."""

    def execute(self, context: OrchestrationContext) -> AgentResult:
        """Generate tests from backend and frontend code."""
        start_time = time.time()
        self.logger.info(f"✅ {self.name}: Generating tests")
        
        backend_code = context.backend_code or ""
        frontend_code = context.frontend_code or ""
        plan = context.plan or {}
        api_endpoints = plan.get("api_endpoints", [])
        
        user_prompt = f"""Generate comprehensive tests for the following application:

Project: {plan.get('project_name', 'Untitled')}

Backend Code Summary:
{backend_code[:2000]}...

Frontend Code Summary:
{frontend_code[:2000]}...

API Endpoints to test:
{json.dumps(api_endpoints, indent=2)}

Generate:
1. Backend tests (pytest) - test all API endpoints and functions
2. Frontend tests (Jest/React Testing Library) - test all components
3. Integration tests
4. Test fixtures and mocks

Provide complete test code for both backend and frontend."""
        
        success, response = self._generate_with_prompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=8192,
            temperature=0.3
        )
        
        if not success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=response,
                execution_time=time.time() - start_time
            )
        
        code = self._extract_code(response)
        
        self.logger.info(f"✅ {self.name}: Generated {len(code)} characters of test code")
        
        return AgentResult(
            agent_name=self.name,
            success=True,
            data=code,
            metadata={
                "code_length": len(code),
                "languages": ["python (pytest)", "javascript (jest)"]
            },
            execution_time=time.time() - start_time
        )
    
    def _extract_code(self, response: str) -> str:
        """Extract code from markdown code blocks."""
        code_pattern = r'```\w*\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return '\n\n'.join(matches)
        
        return response.strip()


class DeploymentAgent(BaseAgent):
    """
    Deployment Agent: Generates deployment configurations and documentation.
    Uses LLM to create Docker files, CI/CD pipelines, and deployment scripts.
    """
    
    SYSTEM_PROMPT = """You are a DevOps/Deployment Agent. Your task is to generate deployment configurations.

Requirements:
- Create production-ready Docker configurations
- Include docker-compose for local development
- Generate CI/CD pipeline configs (GitHub Actions, GitLab CI)
- Include environment variable documentation
- Add health checks and monitoring setup
- Include scaling configurations
- Add SSL/TLS configuration for production
- Write clear deployment documentation

Your response should include all necessary config files and documentation."""

    def execute(self, context: OrchestrationContext) -> AgentResult:
        """Generate deployment configuration."""
        start_time = time.time()
        self.logger.info(f"🚀 {self.name}: Generating deployment configuration")
        
        plan = context.plan or {}
        technologies = plan.get("technologies", {})
        deployment_tech = technologies.get("deployment", ["Docker"])
        
        user_prompt = f"""Generate deployment configuration for the following project:

Project: {plan.get('project_name', 'Untitled')}
Description: {plan.get('description', context.requirement)}
Deployment Technologies: {', '.join(deployment_tech)}

Backend Technologies: {technologies.get('backend', [])}
Frontend Technologies: {technologies.get('frontend', [])}
Database: {technologies.get('database', [])}

Generate:
1. Dockerfile for backend
2. Dockerfile for frontend (if separate)
3. docker-compose.yml for local development
4. docker-compose.prod.yml for production
5. GitHub Actions CI/CD workflow
6. Environment variable template (.env.example)
7. Deployment documentation (README)
8. Kubernetes manifests (optional)

Provide complete, production-ready configuration files."""
        
        success, response = self._generate_with_prompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=8192,
            temperature=0.3
        )
        
        if not success:
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                error=response,
                execution_time=time.time() - start_time
            )
        
        code = self._extract_code(response)
        
        self.logger.info(f"✅ {self.name}: Generated deployment configuration")
        
        return AgentResult(
            agent_name=self.name,
            success=True,
            data=code,
            metadata={
                "code_length": len(code),
                "formats": ["dockerfile", "yaml", "markdown"]
            },
            execution_time=time.time() - start_time
        )
    
    def _extract_code(self, response: str) -> str:
        """Extract code/config from markdown code blocks."""
        code_pattern = r'```\w*\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return '\n\n---\n\n'.join(matches)
        
        return response.strip()


class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow, managing communication and coordination.
    """
    
    def __init__(self, llm: LLMWrapper, output_dir: str = "./generated"):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.agents: Dict[str, BaseAgent] = {}
        self.results: List[AgentResult] = []
        
        # Initialize agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all agents with the shared LLM."""
        self.agents = {
            "planning": PlanningAgent("PlanningAgent", self.llm),
            "backend": BackendAgent("BackendAgent", self.llm),
            "frontend": FrontendAgent("FrontendAgent", self.llm),
            "testing": TestingAgent("TestingAgent", self.llm),
            "deployment": DeploymentAgent("DeploymentAgent", self.llm),
        }
        logger.info(f"🤖 Initialized {len(self.agents)} agents")
    
    def run_workflow(self, requirement: str) -> OrchestrationContext:
        """
        Run the complete multi-agent workflow.
        
        Workflow:
        1. PlanningAgent → Creates implementation plan
        2. BackendAgent → Generates backend code (uses plan)
        3. FrontendAgent → Generates frontend code (uses plan)
        4. TestingAgent → Generates tests (uses backend + frontend)
        5. DeploymentAgent → Creates deployment config (uses all)
        """
        logger.info("=" * 70)
        logger.info("🚀 Starting Multi-Agent Orchestration Workflow")
        logger.info("=" * 70)
        
        context = OrchestrationContext(requirement=requirement)
        
        # Stage 1: Planning
        result = self.agents["planning"].execute(context)
        self.results.append(result)
        
        if result.success:
            context.plan = result.data
            logger.info(f"📋 Plan created: {context.plan.get('project_name', 'Unnamed')}")
            logger.info(f"   Components: {', '.join(context.plan.get('components', []))}")
        else:
            logger.error(f"❌ Planning failed: {result.error}")
            return context
        
        # Stage 2: Backend (depends on plan)
        result = self.agents["backend"].execute(context)
        self.results.append(result)
        
        if result.success:
            context.backend_code = result.data
            logger.info(f"   Backend: {result.metadata.get('code_length', 0)} chars generated")
        else:
            logger.error(f"❌ Backend generation failed: {result.error}")
        
        # Stage 3: Frontend (depends on plan)
        result = self.agents["frontend"].execute(context)
        self.results.append(result)
        
        if result.success:
            context.frontend_code = result.data
            logger.info(f"   Frontend: {result.metadata.get('code_length', 0)} chars generated")
        else:
            logger.error(f"❌ Frontend generation failed: {result.error}")
        
        # Stage 4: Testing (depends on backend + frontend)
        result = self.agents["testing"].execute(context)
        self.results.append(result)
        
        if result.success:
            context.tests = result.data
            logger.info(f"   Tests: {result.metadata.get('code_length', 0)} chars generated")
        else:
            logger.error(f"❌ Test generation failed: {result.error}")
        
        # Stage 5: Deployment (depends on all)
        result = self.agents["deployment"].execute(context)
        self.results.append(result)
        
        if result.success:
            context.deployment_config = result.data
            logger.info(f"   Deployment: Configuration generated")
        else:
            logger.error(f"❌ Deployment generation failed: {result.error}")
        
        # Save all outputs
        self._save_outputs(context)
        
        # Print summary
        self._print_summary()
        
        return context
    
    def _save_outputs(self, context: OrchestrationContext):
        """Save all generated artifacts to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plan
        if context.plan:
            with open(self.output_dir / "plan.json", "w") as f:
                json.dump(context.plan, f, indent=2)
        
        # Save backend code
        if context.backend_code:
            with open(self.output_dir / "backend.py", "w") as f:
                f.write(context.backend_code)
        
        # Save frontend code
        if context.frontend_code:
            with open(self.output_dir / "frontend.tsx", "w") as f:
                f.write(context.frontend_code)
        
        # Save tests
        if context.tests:
            with open(self.output_dir / "tests.py", "w") as f:
                f.write(context.tests)
        
        # Save deployment config
        if context.deployment_config:
            with open(self.output_dir / "deployment.md", "w") as f:
                f.write(context.deployment_config)
        
        logger.info(f"💾 All artifacts saved to: {self.output_dir}")
    
    def _print_summary(self):
        """Print workflow execution summary."""
        logger.info("=" * 70)
        logger.info("📊 Execution Summary")
        logger.info("=" * 70)
        
        total_time = sum(r.execution_time for r in self.results)
        successful = sum(1 for r in self.results if r.success)
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {result.agent_name:20} | {result.execution_time:.2f}s | "
                       f"{'Success' if result.success else 'Failed'}")
        
        logger.info("-" * 70)
        logger.info(f"Total: {successful}/{len(self.results)} agents succeeded | Total time: {total_time:.2f}s")
        logger.info("=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Orchestration with LLM Integration")
    parser.add_argument("--model-path", type=str, default=os.environ.get("NEXUS_MODEL_PATH", "./models"),
                       help="Path to the LLM model")
    parser.add_argument("--requirement", type=str, default="Build a todo app",
                       help="Software requirement to implement")
    parser.add_argument("--output-dir", type=str, default="./generated",
                       help="Directory to save generated artifacts")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--load-in-8bit", action="store_true",
                       help="Load model in 8-bit mode to save memory")
    
    args = parser.parse_args()
    
    # Check environment
    if not check_env():
        print("Warning: Not in 'nexus' conda environment. Continuing anyway...")
    
    # Initialize LLM
    logger.info(f"🔄 Initializing LLM from {args.model_path}")
    llm = LLMWrapper(
        model_path=args.model_path,
        device=args.device,
        load_in_8bit=args.load_in_8bit
    )
    
    if not llm.initialize():
        logger.error("❌ Failed to initialize LLM. Exiting.")
        sys.exit(1)
    
    # Run orchestration
    orchestrator = AgentOrchestrator(llm, output_dir=args.output_dir)
    context = orchestrator.run_workflow(args.requirement)
    
    logger.info("✅ Multi-agent workflow complete!")
    logger.info(f"📁 Generated artifacts saved to: {args.output_dir}")
    
    return context


if __name__ == "__main__":
    main()
