"""
Unit tests for Multi-Agent Orchestration.

Tests the multi-agent system with:
- Planning, Backend, Frontend, Testing, Deployment agents
- Agent orchestration workflow
- Retry logic with exponential backoff
- Error handling and graceful degradation
- Context passing between agents
"""

import pytest
import torch
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestAgentResult:
    """Tests for AgentResult dataclass."""
    
    def test_agent_result_creation(self):
        """Test AgentResult creation."""
        from src.20_multi_agent_orchestration import AgentResult
        
        result = AgentResult(
            agent_name="TestAgent",
            success=True,
            data={"key": "value"},
            metadata={"time": 1.0},
            error=None,
            execution_time=2.5
        )
        
        assert result.agent_name == "TestAgent"
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.execution_time == 2.5


class TestOrchestrationContext:
    """Tests for OrchestrationContext dataclass."""
    
    def test_context_creation(self):
        """Test context creation."""
        from src.20_multi_agent_orchestration import OrchestrationContext
        
        context = OrchestrationContext(
            requirement="Build a todo app",
            plan={"name": "TodoApp"}
        )
        
        assert context.requirement == "Build a todo app"
        assert context.plan == {"name": "TodoApp"}
        assert context.backend_code is None
        assert context.frontend_code is None


class TestLLMWrapper:
    """Tests for LLMWrapper class."""
    
    @pytest.fixture
    def mock_model_tokenizer(self):
        """Create mock model and tokenizer."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        
        # Mock generate output
        mock_output = torch.randint(0, 1000, (1, 20))
        mock_model.generate.return_value = mock_output
        
        # Mock tokenizer decode
        mock_tokenizer.decode.return_value = "Generated response"
        
        return mock_model, mock_tokenizer
    
    @pytest.fixture
    def llm_wrapper(self):
        """Create LLMWrapper with mocked dependencies."""
        with patch.dict('sys.modules', {'src.omni.loader': MagicMock()}):
            from src.20_multi_agent_orchestration import LLMWrapper
            return LLMWrapper(
                model_path="/fake/path",
                device="cpu",
                load_in_8bit=False
            )
    
    def test_initialization(self, llm_wrapper):
        """Test LLMWrapper initialization."""
        assert llm_wrapper.model_path == "/fake/path"
        assert llm_wrapper.device == "cpu"
        assert llm_wrapper.load_in_8bit is False
        assert llm_wrapper._initialized is False
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_initialize_success(self, mock_loader_class, llm_wrapper, mock_model_tokenizer):
        """Test successful LLM initialization."""
        mock_model, mock_tokenizer = mock_model_tokenizer
        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        success = llm_wrapper.initialize()
        
        assert success is True
        assert llm_wrapper._initialized is True
        assert llm_wrapper.model is mock_model
        assert llm_wrapper.tokenizer is mock_tokenizer
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_initialize_failure(self, mock_loader_class, llm_wrapper):
        """Test LLM initialization failure."""
        mock_loader = MagicMock()
        mock_loader.load.side_effect = Exception("Model not found")
        mock_loader_class.return_value = mock_loader
        
        success = llm_wrapper.initialize()
        
        assert success is False
        assert llm_wrapper._initialized is False
    
    def test_initialize_already_initialized(self, llm_wrapper):
        """Test initialization when already initialized."""
        llm_wrapper._initialized = True
        
        success = llm_wrapper.initialize()
        
        assert success is True
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_generate_success(self, mock_loader_class, llm_wrapper, mock_model_tokenizer):
        """Test successful text generation."""
        mock_model, mock_tokenizer = mock_model_tokenizer
        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        llm_wrapper.initialize()
        success, text = llm_wrapper.generate("Test prompt")
        
        assert success is True
        assert text == "Generated response"
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_generate_with_stop_sequences(self, mock_loader_class, llm_wrapper, mock_model_tokenizer):
        """Test generation with stop sequences."""
        mock_model, mock_tokenizer = mock_model_tokenizer
        mock_tokenizer.decode.return_value = "Response stops here Human: more text"
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        llm_wrapper.initialize()
        success, text = llm_wrapper.generate(
            "Test prompt",
            stop_sequences=["Human:"]
        )
        
        assert success is True
        assert "Human:" not in text
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_generate_retry_success(self, mock_loader_class, llm_wrapper, mock_model_tokenizer):
        """Test generation with retry that eventually succeeds."""
        mock_model, mock_tokenizer = mock_model_tokenizer
        
        # First call fails, second succeeds
        mock_model.generate.side_effect = [
            Exception("First attempt failed"),
            torch.randint(0, 1000, (1, 20))
        ]
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        llm_wrapper.initialize()
        
        with patch('time.sleep'):  # Speed up test
            success, text = llm_wrapper.generate("Test prompt", retries=2)
        
        assert success is True
        assert mock_model.generate.call_count == 2
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_generate_all_retries_fail(self, mock_loader_class, llm_wrapper, mock_model_tokenizer):
        """Test generation when all retries fail."""
        mock_model, mock_tokenizer = mock_model_tokenizer
        mock_model.generate.side_effect = Exception("Always fails")
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        llm_wrapper.initialize()
        
        with patch('time.sleep'):  # Speed up test
            success, text = llm_wrapper.generate("Test prompt", retries=3)
        
        assert success is False
        assert "Failed after 3 attempts" in text
        assert mock_model.generate.call_count == 3
    
    def test_generate_without_initialization(self, llm_wrapper):
        """Test generation without initialization."""
        success, text = llm_wrapper.generate("Test prompt")
        
        assert success is False
        assert "Failed to initialize LLM" in text


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""
    
    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with patch('src.20_multi_agent_orchestration.OmniModelLoader'):
            from src.20_multi_agent_orchestration import BaseAgent, LLMWrapper
            
            llm = LLMWrapper(model_path="/fake")
            
            with pytest.raises(TypeError):
                BaseAgent("TestAgent", llm)


class TestPlanningAgent:
    """Tests for PlanningAgent."""
    
    @pytest.fixture
    def planning_agent(self):
        """Create PlanningAgent with mocked LLM."""
        with patch('src.20_multi_agent_orchestration.OmniModelLoader'):
            from src.20_multi_agent_orchestration import PlanningAgent, LLMWrapper
            
            mock_llm = MagicMock()
            agent = PlanningAgent("PlanningAgent", mock_llm)
            return agent
    
    def test_execute_success(self, planning_agent):
        """Test successful plan generation."""
        mock_response = json.dumps({
            "project_name": "TodoApp",
            "description": "A todo app",
            "components": ["backend", "frontend"],
            "architecture": "MVC",
            "technologies": {
                "backend": ["FastAPI"],
                "frontend": ["React"],
                "database": ["PostgreSQL"],
                "deployment": ["Docker"]
            },
            "api_endpoints": [],
            "data_models": [],
            "ui_components": [],
            "timeline": "2 weeks",
            "milestones": ["MVP", "Production"]
        })
        
        planning_agent.llm.generate.return_value = (True, mock_response)
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(requirement="Build a todo app")
        
        result = planning_agent.execute(context)
        
        assert result.success is True
        assert result.data["project_name"] == "TodoApp"
        assert result.metadata["components_count"] == 2
    
    def test_execute_with_extra_text(self, planning_agent):
        """Test parsing JSON from response with extra text."""
        mock_response = """
        Here's the plan:
        ```json
        {"project_name": "Test", "components": [], "architecture": "test", 
         "technologies": {}, "api_endpoints": [], "data_models": [], 
         "ui_components": [], "timeline": "1 week", "milestones": []}
        ```
        """
        planning_agent.llm.generate.return_value = (True, mock_response)
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(requirement="Build something")
        
        result = planning_agent.execute(context)
        
        assert result.success is True
    
    def test_execute_json_parse_error(self, planning_agent):
        """Test handling of JSON parse error."""
        planning_agent.llm.generate.return_value = (True, "Not valid JSON")
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(requirement="Build something")
        
        result = planning_agent.execute(context)
        
        assert result.success is False
        assert "JSON parse error" in result.error
    
    def test_execute_generation_failure(self, planning_agent):
        """Test handling of generation failure."""
        planning_agent.llm.generate.return_value = (False, "Model error")
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(requirement="Build something")
        
        result = planning_agent.execute(context)
        
        assert result.success is False
        assert result.error == "Model error"


class TestBackendAgent:
    """Tests for BackendAgent."""
    
    @pytest.fixture
    def backend_agent(self):
        """Create BackendAgent with mocked LLM."""
        mock_llm = MagicMock()
        
        from src.20_multi_agent_orchestration import BackendAgent
        return BackendAgent("BackendAgent", mock_llm)
    
    def test_execute_success(self, backend_agent):
        """Test successful backend code generation."""
        mock_response = """
        ```python
        from fastapi import FastAPI
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "Hello World"}
        ```
        """
        backend_agent.llm.generate.return_value = (True, mock_response)
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(
            requirement="Build API",
            plan={
                "project_name": "TestAPI",
                "api_endpoints": [{"method": "GET", "path": "/", "description": "Root"}],
                "technologies": {"backend": ["FastAPI"]}
            }
        )
        
        result = backend_agent.execute(context)
        
        assert result.success is True
        assert "FastAPI" in result.data
        assert result.metadata["language"] == "python"
    
    def test_code_extraction(self, backend_agent):
        """Test code extraction from markdown."""
        response_with_code = "```python\ncode here\n```"
        extracted = backend_agent._extract_code(response_with_code)
        
        assert "code here" in extracted
    
    def test_code_extraction_no_markers(self, backend_agent):
        """Test code extraction without markdown markers."""
        response_plain = "Plain text response"
        extracted = backend_agent._extract_code(response_plain)
        
        assert extracted == response_plain.strip()


class TestFrontendAgent:
    """Tests for FrontendAgent."""
    
    @pytest.fixture
    def frontend_agent(self):
        """Create FrontendAgent with mocked LLM."""
        mock_llm = MagicMock()
        
        from src.20_multi_agent_orchestration import FrontendAgent
        return FrontendAgent("FrontendAgent", mock_llm)
    
    def test_execute_success(self, frontend_agent):
        """Test successful frontend code generation."""
        mock_response = """
        ```typescript
        import React from 'react';
        const App = () => <div>Hello</div>;
        export default App;
        ```
        """
        frontend_agent.llm.generate.return_value = (True, mock_response)
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(
            requirement="Build UI",
            plan={
                "project_name": "TestUI",
                "ui_components": ["Header", "Footer"],
                "technologies": {"frontend": ["React"]}
            }
        )
        
        result = frontend_agent.execute(context)
        
        assert result.success is True
        assert "React" in result.data


class TestTestingAgent:
    """Tests for TestingAgent."""
    
    @pytest.fixture
    def testing_agent(self):
        """Create TestingAgent with mocked LLM."""
        mock_llm = MagicMock()
        
        from src.20_multi_agent_orchestration import TestingAgent
        return TestingAgent("TestingAgent", mock_llm)
    
    def test_execute_success(self, testing_agent):
        """Test successful test generation."""
        mock_response = """
        ```python
        def test_example():
            assert True
        ```
        """
        testing_agent.llm.generate.return_value = (True, mock_response)
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(
            requirement="Build app",
            plan={"project_name": "TestApp"},
            backend_code="def api(): pass",
            frontend_code="const App = () => {}"
        )
        
        result = testing_agent.execute(context)
        
        assert result.success is True
        assert "python" in result.metadata["languages"]


class TestDeploymentAgent:
    """Tests for DeploymentAgent."""
    
    @pytest.fixture
    def deployment_agent(self):
        """Create DeploymentAgent with mocked LLM."""
        mock_llm = MagicMock()
        
        from src.20_multi_agent_orchestration import DeploymentAgent
        return DeploymentAgent("DeploymentAgent", mock_llm)
    
    def test_execute_success(self, deployment_agent):
        """Test successful deployment config generation."""
        mock_response = """
        ```dockerfile
        FROM python:3.9
        COPY . /app
        ```
        
        ```yaml
        version: '3'
        services:
          app:
            build: .
        ```
        """
        deployment_agent.llm.generate.return_value = (True, mock_response)
        
        from src.20_multi_agent_orchestration import OrchestrationContext
        context = OrchestrationContext(
            requirement="Deploy app",
            plan={
                "project_name": "TestApp",
                "technologies": {
                    "backend": ["FastAPI"],
                    "frontend": ["React"],
                    "deployment": ["Docker"]
                }
            }
        )
        
        result = deployment_agent.execute(context)
        
        assert result.success is True
        assert "dockerfile" in result.metadata["formats"]


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create orchestrator with mocked LLM."""
        mock_llm = MagicMock()
        
        from src.20_multi_agent_orchestration import AgentOrchestrator
        return AgentOrchestrator(llm=mock_llm, output_dir=str(tmp_path / "output"))
    
    def test_initialization(self, orchestrator, tmp_path):
        """Test orchestrator initialization."""
        assert orchestrator.llm is not None
        assert len(orchestrator.agents) == 5
        assert "planning" in orchestrator.agents
        assert "backend" in orchestrator.agents
        assert "frontend" in orchestrator.agents
        assert "testing" in orchestrator.agents
        assert "deployment" in orchestrator.agents
    
    def test_run_workflow_success(self, orchestrator):
        """Test successful workflow execution."""
        # Mock all agents
        for agent in orchestrator.agents.values():
            agent.execute = MagicMock(return_value=MagicMock(
                success=True,
                data={"test": "data"},
                metadata={},
                execution_time=1.0
            ))
        
        context = orchestrator.run_workflow("Build a todo app")
        
        assert context.requirement == "Build a todo app"
        assert all(r.success for r in orchestrator.results)
    
    def test_run_workflow_planning_failure(self, orchestrator):
        """Test workflow when planning fails."""
        # Make planning fail
        orchestrator.agents["planning"].execute = MagicMock(return_value=MagicMock(
            success=False,
            data=None,
            error="Planning failed",
            execution_time=1.0
        ))
        
        context = orchestrator.run_workflow("Build app")
        
        assert context.plan is None
        # Other agents should still run but with empty context
    
    def test_context_passing(self, orchestrator):
        """Test context is passed correctly between agents."""
        results = []
        
        def capture_context(agent_name):
            def execute(context):
                results.append((agent_name, context))
                return MagicMock(success=True, data={"agent": agent_name}, execution_time=0.1)
            return execute
        
        for name, agent in orchestrator.agents.items():
            agent.execute = MagicMock(side_effect=capture_context(name))
        
        orchestrator.run_workflow("Test")
        
        # Verify each agent received context
        assert len(results) == 5
    
    def test_save_outputs(self, orchestrator, tmp_path):
        """Test output saving."""
        from src.20_multi_agent_orchestration import OrchestrationContext
        
        context = OrchestrationContext(
            requirement="Test",
            plan={"name": "Test"},
            backend_code="def api(): pass",
            frontend_code="const App = () => {}",
            tests="def test(): pass",
            deployment_config="# Dockerfile"
        )
        
        orchestrator._save_outputs(context)
        
        assert (orchestrator.output_dir / "plan.json").exists()
        assert (orchestrator.output_dir / "backend.py").exists()
        assert (orchestrator.output_dir / "frontend.tsx").exists()
        assert (orchestrator.output_dir / "tests.py").exists()
        assert (orchestrator.output_dir / "deployment.md").exists()
    
    def test_print_summary(self, orchestrator, caplog):
        """Test summary printing."""
        orchestrator.results = [
            MagicMock(agent_name="Agent1", success=True, execution_time=1.0),
            MagicMock(agent_name="Agent2", success=False, execution_time=2.0)
        ]
        
        with caplog.at_level("INFO"):
            orchestrator._print_summary()
        
        assert "Execution Summary" in caplog.text
        assert "Agent1" in caplog.text


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""
    
    @patch('src.20_multi_agent_orchestration.OmniModelLoader')
    def test_exponential_backoff_timing(self, mock_loader_class):
        """Test that exponential backoff waits correct amounts."""
        from src.20_multi_agent_orchestration import LLMWrapper
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.decode.return_value = "Response"
        
        # Fail twice, succeed on third
        mock_model.generate.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            torch.randint(0, 1000, (1, 10))
        ]
        
        mock_loader = MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        llm = LLMWrapper("/fake")
        llm.initialize()
        
        sleep_times = []
        with patch('time.sleep', side_effect=lambda x: sleep_times.append(x)):
            llm.generate("Test", retries=3)
        
        # Exponential backoff: 2^0, 2^1 = 1, 2 seconds
        assert sleep_times == [1.0, 2.0]


class TestErrorHandling:
    """Tests for error handling and graceful degradation."""
    
    def test_check_env_correct_env(self, monkeypatch):
        """Test environment check in correct env."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "nexus")
        
        from src.20_multi_agent_orchestration import check_env
        assert check_env() is True
    
    def test_check_env_wrong_env(self, monkeypatch):
        """Test environment check in wrong env."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "other")
        
        from src.20_multi_agent_orchestration import check_env
        assert check_env() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
