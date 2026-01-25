"""
Unit tests for Fullstack evaluation benchmark.
(MOCKED)
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.benchmarks.fullstack_eval import (
    FullstackEval, EvalCase, EvalResult,
    RESTApiEvaluator, SQLEvaluator, ReactAccessibilityEvaluator,
    KubernetesEvaluator, TerraformEvaluator, CICDEvaluator
)

class TestFullstackEval:
    @pytest.fixture
    def runner(self):
        return FullstackEval()

    def test_dummy_model(self, runner):
        assert "Response to:" in runner._dummy_model("test")

    def test_api_evaluator(self):
        evaluator = RESTApiEvaluator()
        case = evaluator.get_cases()[0]
        # Hit all criteria
        resp = "POST /users validation schema 201 created success error 400 404 422 500 jwt auth password"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0
        assert len(res.passed_criteria) > 0

    def test_api_evaluator_pagination(self):
        evaluator = RESTApiEvaluator()
        case = evaluator.get_cases()[1]
        resp = "GET page limit offset cursor next filter category price search sort order asc desc data items total meta index cache efficient"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_api_evaluator_cart(self):
        evaluator = RESTApiEvaluator()
        case = evaluator.get_cases()[2]
        resp = "POST /cart checkout concurrent lock optimistic version etag idempoten retry key"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_sql_evaluator(self):
        evaluator = SQLEvaluator()
        case = evaluator.get_cases()[0]
        resp = "create table users primary key foreign key references references index create index varchar int join sum count group by order by 30 day interval explain efficient"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_sql_evaluator_migration(self):
        evaluator = SQLEvaluator()
        case = evaluator.get_cases()[2]
        resp = "add column null default batch update where not null constraint alter rollback drop no lock online"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_react_evaluator(self):
        evaluator = ReactAccessibilityEvaluator()
        case = evaluator.get_cases()[0]
        resp = "aria- role=dialog onkeydown escape focus ref <dialog props htmlfor error required <form"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_k8s_evaluator(self):
        evaluator = KubernetesEvaluator()
        case = evaluator.get_cases()[0]
        resp = "kind: deployment resources: limits: livenessprobe hpa securitycontext labels: statefulset persistentvolumeclaim secret volume config backup"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_terraform_evaluator(self):
        evaluator = TerraformEvaluator()
        case = evaluator.get_cases()[0]
        resp = "aws_vpc aws_subnet public nat_gateway route_table variable var. tags module aws_db_instance multi_az encrypt kms backup cloudwatch"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_cicd_evaluator(self):
        evaluator = CICDEvaluator()
        case = evaluator.get_cases()[0]
        resp = "name: jobs: npm run build npm run test cache secrets. deploy staging approval rollback environment: canary"
        res = evaluator.evaluate_response(case, resp)
        assert res.score > 0

    def test_run_category(self, runner):
        res = runner.run_category("api")
        assert res["category"] == "api"
        assert len(res["results"]) > 0

    def test_run_category_invalid(self, runner):
        with pytest.raises(ValueError):
            runner.run_category("invalid")

    def test_run_all(self, runner):
        res = runner.run_all()
        assert "overall_score" in res

    def test_save_results(self, runner, tmp_path):
        out = tmp_path / "res.json"
        runner.save_results(out, {"test": 1})
        assert out.exists()

    def test_get_all_cases(self, runner):
        assert len(runner.get_all_cases()) > 0

    def test_main_eval_all(self, tmp_path):
        out = tmp_path / "fullstack.json"
        with patch("sys.argv", ["fullstack.py", "--eval", "all", "--output", str(out)]):
            from src.benchmarks.fullstack_eval import main
            main()
            assert out.exists()

    def test_main_list_cases(self):
        with patch("sys.argv", ["fullstack.py", "--list-cases"]):
            from src.benchmarks.fullstack_eval import main
            main()
