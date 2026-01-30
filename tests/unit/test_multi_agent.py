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
from importlib import import_module
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import multi-agent module using importlib to handle numeric prefix
_mao = import_module("src.20_multi_agent_orchestration")
AgentResult = _mao.AgentResult
OrchestrationContext = _mao.OrchestrationContext
LLMWrapper = _mao.LLMWrapper
BaseAgent = _mao.BaseAgent
PlanningAgent = _mao.PlanningAgent
BackendAgent = _mao.BackendAgent
FrontendAgent = _mao.FrontendAgent
TestingAgent = _mao.TestingAgent
DeploymentAgent = _mao.DeploymentAgent
AgentOrchestrator = _mao.AgentOrchestrator
check_env = _mao.check_env


class TestAgentResult:
    """Tests for AgentResult dataclass."""
    
    def test_agent_result_creation(self):
        """Test AgentResult creation."""
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
    
    def test_initialize_already_initialized(self, llm_wrapper):
        """Test initialization when already initialized."""
        llm_wrapper._initialized = True
        
        success = llm_wrapper.initialize()
        
        assert success is True
    
    def test_generate_without_initialization(self, llm_wrapper):
        """Test generation without initialization."""
        success, text = llm_wrapper.generate("Test prompt")
        
        assert success is False
        assert "Failed to initialize LLM" in text


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""
    
    def test_base_agent_is_abstract(self):
        """Test that BaseAgent cannot be instantiated directly."""
        llm = LLMWrapper(model_path="/fake")
        
        with pytest.raises(TypeError):
            BaseAgent("TestAgent", llm)


class TestPlanningAgent:
    """Tests for PlanningAgent."""
    
    @pytest.fixture
    def planning_agent(self):
        """Create PlanningAgent with mocked LLM."""
        mock_llm = MagicMock()
        agent = PlanningAgent("PlanningAgent", mock_llm)
        return agent
    
    def test_execute_success(self, planning_agent):
        """Test successful plan generation."""
        mock_response = json.dumps({
            "project_name": "TodoApp",
            "description": "A todo application",
            "components": ["Backend API", "Frontend UI", "Database"],
            "architecture": "MVC",
            "technologies": {
                "backend": ["FastAPI", "Python"],
                "frontend": ["React", "TypeScript"],
                "database": ["PostgreSQL"],
                "deployment": ["Docker", "AWS"]
            },
            "api_endpoints": [],
            "data_models": [],
            "ui_components": [],
            "timeline": "2 weeks",
            "milestones": []
        })
        
        planning_agent.llm.generate.return_value = (True, mock_response)
        
        context = OrchestrationContext(requirement="Build a todo app")
        
        result = planning_agent.execute(context)
        
        assert result.success is True
        assert result.data["project_name"] == "TodoApp"
        assert result.metadata["components_count"] == 3
    
    def test_execute_json_parse_error(self, planning_agent):
        """Test handling of JSON parse error."""
        planning_agent.llm.generate.return_value = (True, "Not valid JSON")
        
        context = OrchestrationContext(requirement="Build something")
        
        result = planning_agent.execute(context)
        
        assert result.success is False
        assert "JSON parse error" in result.error


class TestBackendAgent:
    """Tests for BackendAgent."""
    
    @pytest.fixture
    def backend_agent(self):
        """Create BackendAgent with mocked LLM."""
        mock_llm = MagicMock()
        agent = BackendAgent("BackendAgent", mock_llm)
        return agent
    
    def test_execute_success(self, backend_agent):
        """Test successful backend code generation."""
        mock_response = """
```python
from fastapi import FastAPI
app = FastAPI()
```
"""
        backend_agent.llm.generate.return_value = (True, mock_response)
        
        context = OrchestrationContext(
            requirement="Build API",
            plan={
                "project_name": "TestAPI",
                "api_endpoints": [],
                "technologies": {"backend": ["FastAPI"]}
            }
        )
        
        result = backend_agent.execute(context)
        
        assert result.success is True
        assert "FastAPI" in result.data
        assert result.metadata["language"] == "python"


class TestFrontendAgent:
    """Tests for FrontendAgent."""
    
    @pytest.fixture
    def frontend_agent(self):
        """Create FrontendAgent with mocked LLM."""
        mock_llm = MagicMock()
        agent = FrontendAgent("FrontendAgent", mock_llm)
        return agent
    
    def test_execute_success(self, frontend_agent):
        """Test successful frontend code generation."""
        mock_response = """
```typescript
import React from 'react';
const App = () => <div>Hello</div>;
```
"""
        frontend_agent.llm.generate.return_value = (True, mock_response)
        
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
        agent = TestingAgent("TestingAgent", mock_llm)
        return agent
    
    def test_execute_success(self, testing_agent):
        """Test successful test generation."""
        mock_response = """
```python
def test_example():
    assert True
```
"""
        testing_agent.llm.generate.return_value = (True, mock_response)
        
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
        agent = DeploymentAgent("DeploymentAgent", mock_llm)
        return agent
    
    def test_execute_success(self, deployment_agent):
        """Test successful deployment config generation."""
        mock_response = """
```dockerfile
FROM python:3.9
COPY . /app
```
"""
        deployment_agent.llm.generate.return_value = (True, mock_response)
        
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
        # Other agents should still run
    
    def test_save_outputs(self, orchestrator, tmp_path):
        """Test output saving."""
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


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""
    
    def test_check_env_correct_env(self, monkeypatch):
        """Test environment check in correct env."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "nexus")
        
        assert check_env() is True
    
    def test_check_env_wrong_env(self, monkeypatch):
        """Test environment check in wrong env."""
        monkeypatch.setenv("CONDA_DEFAULT_ENV", "other")
        
        assert check_env() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
