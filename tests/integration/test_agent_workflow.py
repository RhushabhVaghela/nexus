"""
Integration tests for Multi-Agent Workflow.

Tests the full multi-agent development workflow including:
- Complete workflow from requirements to deployment
- Agent communication and coordination
- Output artifact generation and validation
- Error recovery and workflow continuation
"""

import pytest
import torch
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, call
from importlib import import_module
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import multi-agent module using importlib
_mao = import_module("src.20_multi_agent_orchestration")
AgentOrchestrator = _mao.AgentOrchestrator
PlanningAgent = _mao.PlanningAgent
BackendAgent = _mao.BackendAgent
FrontendAgent = _mao.FrontendAgent
TestingAgent = _mao.TestingAgent
DeploymentAgent = _mao.DeploymentAgent
OrchestrationContext = _mao.OrchestrationContext
LLMWrapper = _mao.LLMWrapper
AgentResult = _mao.AgentResult


@pytest.mark.integration
class TestAgentWorkflow:
    """Integration tests for full agent workflow."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM with realistic responses."""
        mock = MagicMock()
        
        responses = {
            "planning": json.dumps({
                "project_name": "TodoApp",
                "description": "A simple todo application",
                "components": ["Backend API", "Frontend UI", "Database"],
                "architecture": "MVC",
                "technologies": {
                    "backend": ["FastAPI", "Python"],
                    "frontend": ["React", "TypeScript"],
                    "database": ["PostgreSQL"],
                    "deployment": ["Docker", "AWS"]
                },
                "api_endpoints": [
                    {"method": "GET", "path": "/todos", "description": "List all todos"},
                    {"method": "POST", "path": "/todos", "description": "Create new todo"}
                ],
                "data_models": [
                    {"name": "Todo", "fields": ["id: int", "title: str", "completed: bool"]}
                ],
                "ui_components": ["TodoList", "TodoForm", "TodoItem"],
                "timeline": "2 weeks",
                "milestones": ["Backend API", "Frontend UI", "Deployment"]
            }),
            "backend": "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n```",
            "frontend": "```typescript\nimport React from 'react';\nconst App = () => <div>Hello</div>;\n```",
            "testing": "```python\ndef test_example():\n    pass\n```",
            "deployment": "```dockerfile\nFROM python:3.9\nCOPY . /app\n```"
        }
        
        def generate(prompt, **kwargs):
            if "Planning Agent" in prompt or "implementation plan" in prompt.lower():
                return True, responses["planning"]
            elif "Backend Development Agent" in prompt or "backend code" in prompt.lower():
                return True, responses["backend"]
            elif "Frontend Development Agent" in prompt or "frontend code" in prompt.lower():
                return True, responses["frontend"]
            elif "Testing Agent" in prompt or "test suites" in prompt.lower():
                return True, responses["testing"]
            elif "Deployment Agent" in prompt or "deployment configuration" in prompt.lower():
                return True, responses["deployment"]
            else:
                return True, "Default response"
        
        mock.generate = Mock(side_effect=generate)
        return mock
    
    @pytest.fixture
    def orchestrator(self, mock_llm, tmp_path):
        """Create orchestrator with mocked LLM."""
        return AgentOrchestrator(llm=mock_llm, output_dir=str(tmp_path / "output"))
    
    def test_complete_workflow(self, orchestrator, tmp_path):
        """Test complete workflow from requirement to deployment."""
        requirement = "Build a todo app"
        
        context = orchestrator.run_workflow(requirement)
        
        # Verify all artifacts were created
        assert context.requirement == requirement
        assert context.plan is not None
        assert context.backend_code is not None
        assert context.frontend_code is not None
        assert context.tests is not None
        assert context.deployment_config is not None
        
        # Verify all agents succeeded
        assert len(orchestrator.results) == 5
        assert all(r.success for r in orchestrator.results)
        
        print("✅ Complete workflow executed successfully")
    
    def test_artifact_generation(self, orchestrator, tmp_path):
        """Test that all artifacts are generated and saved."""
        requirement = "Build a todo app"
        context = orchestrator.run_workflow(requirement)
        
        # Verify files were saved
        output_dir = orchestrator.output_dir
        assert (output_dir / "plan.json").exists()
        assert (output_dir / "backend.py").exists()
        assert (output_dir / "frontend.tsx").exists()
        assert (output_dir / "tests.py").exists()
        assert (output_dir / "deployment.md").exists()
        
        # Verify content
        with open(output_dir / "plan.json") as f:
            plan = json.load(f)
            assert plan["project_name"] == "TodoApp"
        
        with open(output_dir / "backend.py") as f:
            backend = f.read()
            assert "FastAPI" in backend
        
        print("✅ All artifacts generated and saved correctly")
    
    def test_agent_execution_order(self, orchestrator):
        """Test that agents execute in correct order."""
        requirement = "Build a todo app"
        
        context = orchestrator.run_workflow(requirement)
        
        # Verify execution order
        expected_order = ["PlanningAgent", "BackendAgent", "FrontendAgent", "TestingAgent", "DeploymentAgent"]
        actual_order = [r.agent_name for r in orchestrator.results]
        
        assert actual_order == expected_order
        
        print("✅ Agents executed in correct order")
    
    def test_context_passing_between_agents(self, orchestrator):
        """Test that context is correctly passed between agents."""
        requirement = "Build a todo app"
        
        context = orchestrator.run_workflow(requirement)
        
        # Planning creates the plan
        assert context.plan is not None
        
        # Backend uses the plan
        assert "Todo" in context.backend_code
        
        # Frontend uses the plan
        assert "App" in context.frontend_code
        
        # Testing uses backend and frontend
        assert context.tests is not None
        
        # Deployment uses everything
        assert context.deployment_config is not None
        
        print("✅ Context passed correctly between agents")


@pytest.mark.integration
class TestAgentCommunication:
    """Integration tests for agent communication."""
    
    def test_agent_metadata_tracking(self):
        """Test that agents track metadata correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (True, json.dumps({
            "project_name": "Test",
            "components": ["A", "B"],
            "architecture": "test",
            "technologies": {},
            "api_endpoints": [],
            "data_models": [],
            "ui_components": [],
            "timeline": "1 week",
            "milestones": []
        }))
        
        agent = PlanningAgent("PlanningAgent", mock_llm)
        context = OrchestrationContext(requirement="Test")
        
        result = agent.execute(context)
        
        assert result.success is True
        assert "components_count" in result.metadata
        assert result.execution_time > 0
        
        print("✅ Agent metadata tracked correctly")
    
    def test_error_propagation(self):
        """Test error handling and propagation."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (False, "LLM Error")
        
        agent = PlanningAgent("PlanningAgent", mock_llm)
        context = OrchestrationContext(requirement="Test")
        
        result = agent.execute(context)
        
        assert result.success is False
        assert result.error == "LLM Error"
        assert result.data is None
        
        print("✅ Errors propagated correctly")


@pytest.mark.integration
class TestOutputValidation:
    """Integration tests for output validation."""
    
    def test_json_output_validation(self):
        """Test that agent outputs valid JSON."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (True, json.dumps({
            "project_name": "Test",
            "components": [],
            "architecture": "test",
            "technologies": {},
            "api_endpoints": [],
            "data_models": [],
            "ui_components": [],
            "timeline": "1 week",
            "milestones": []
        }))
        
        agent = PlanningAgent("PlanningAgent", mock_llm)
        context = OrchestrationContext(requirement="Test")
        
        result = agent.execute(context)
        
        assert result.success is True
        json_str = json.dumps(result.data)
        parsed = json.loads(json_str)
        assert parsed["project_name"] == "Test"
        
        print("✅ JSON output validated correctly")
    
    def test_code_extraction_from_markdown(self):
        """Test code extraction from markdown code blocks."""
        agent = BackendAgent("BackendAgent", MagicMock())
        
        response1 = "```python\ncode1\n```"
        assert "code1" in agent._extract_code(response1)
        
        response2 = "```\ncode2\n```"
        assert "code2" in agent._extract_code(response2)
        
        response3 = "plain text"
        assert agent._extract_code(response3) == "plain text"
        
        print("✅ Code extraction from markdown works correctly")


@pytest.mark.integration
class TestWorkflowPerformance:
    """Integration tests for workflow performance."""
    
    def test_total_execution_time(self, tmp_path):
        """Test that workflow completes in reasonable time."""
        import time
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (True, json.dumps({
            "project_name": "Test",
            "components": [],
            "architecture": "test",
            "technologies": {},
            "api_endpoints": [],
            "data_models": [],
            "ui_components": [],
            "timeline": "1 week",
            "milestones": []
        }))
        
        orchestrator = AgentOrchestrator(llm=mock_llm, output_dir=str(tmp_path / "output"))
        
        start_time = time.time()
        context = orchestrator.run_workflow("Test requirement")
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Should complete quickly with mocked LLM
        assert total_time < 10
        assert len(orchestrator.results) == 5
        
        print(f"✅ Workflow completed in {total_time:.2f} seconds")
    
    def test_individual_agent_execution_time(self):
        """Test that individual agents track execution time."""
        import time
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (True, json.dumps({
            "project_name": "Test",
            "components": [],
            "architecture": "test",
            "technologies": {},
            "api_endpoints": [],
            "data_models": [],
            "ui_components": [],
            "timeline": "1 week",
            "milestones": []
        }))
        
        agent = PlanningAgent("PlanningAgent", mock_llm)
        context = OrchestrationContext(requirement="Test")
        
        result = agent.execute(context)
        
        assert result.execution_time > 0
        assert result.execution_time < 5
        
        print(f"✅ Agent execution time tracked: {result.execution_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
