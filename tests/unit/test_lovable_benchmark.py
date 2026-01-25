"""
Unit tests for Lovable benchmark.
(MOCKED)
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.benchmarks.lovable_benchmark import LovableBenchmark, UIGenCase, UIGenResult

class TestLovableBenchmark:
    @pytest.fixture
    def lovable_bench(self):
        return LovableBenchmark()

    def test_dummy_model(self, lovable_bench):
        resp = lovable_bench._dummy_model("test prompt")
        assert "Generated code" in resp

    def test_evaluate_response_basic(self, lovable_bench):
        case = UIGenCase(
            id="test",
            category="screenshot_to_code",
            description="desc",
            prompt="prompt",
            expected_files=["App.tsx"],
            required_elements={"App.tsx": ["button"]},
            rubric={"present": 1, "typescript": 1},
            difficulty="easy"
        )
        
        # Test with high-quality response, include filename
        response = "interface Props {}; <button>Click</button> // App.tsx content substantial " + "x" * 600
        result = lovable_bench.evaluate_response(case, response)
        assert result.score > 0
        assert "App.tsx" in result.files_generated
        assert result.requirements_met["App.tsx:button"] is True

    def test_evaluate_response_rubrics(self, lovable_bench):
        case = UIGenCase(
            id="t2", category="c", description="d", prompt="p",
            expected_files=["f.ts"], required_elements={},
            rubric={
                "loading": 1, "error": 1, "accessibility": 1, 
                "responsive": 1, "variant": 1, "optimistic": 1,
                "generic": 1
            },
            difficulty="e"
        )
        # Hit all keywords
        resp = "loading spinner error catch aria-role md: variant optimistic rollback " + "y" * 600
        result = lovable_bench.evaluate_response(case, resp)
        # max score is 7
        assert result.score == 7

    def test_run_category(self, lovable_bench):
        with patch.object(lovable_bench, 'model_fn', return_value="some response"):
            res = lovable_bench.run_category("screenshot_to_code")
            assert res["category"] == "screenshot_to_code"
            assert len(res["results"]) > 0

    def test_run_category_invalid(self, lovable_bench):
        with pytest.raises(ValueError):
            lovable_bench.run_category("invalid")

    def test_run_all(self, lovable_bench):
        with patch.object(lovable_bench, 'run_category', return_value={"total_score": 1, "max_score": 1, "results": [], "percentage": 100}):
            res = lovable_bench.run_all()
            assert "overall_score" in res

    def test_save_results(self, lovable_bench, tmp_path):
        out = tmp_path / "res.json"
        lovable_bench.save_results(out, {"test": 1})
        assert out.exists()

    def test_export_prompts(self, lovable_bench, tmp_path):
        out = tmp_path / "prompts.json"
        lovable_bench.export_prompts(out)
        assert out.exists()

    def test_get_all_cases(self, lovable_bench):
        cases = lovable_bench.get_all_cases()
        assert len(cases) > 0

    def test_main_eval_all(self, tmp_path):
        out = tmp_path / "main_res.json"
        with patch("sys.argv", ["lovable.py", "--eval", "all", "--output", str(out)]), \
             patch("src.benchmarks.lovable_benchmark.LovableBenchmark.run_all", return_value={"overall_score": 0, "overall_max": 0, "overall_percentage": 0, "categories": {}}):
            from src.benchmarks.lovable_benchmark import main
            main()
            assert out.exists()

    def test_main_list_cases(self):
        with patch("sys.argv", ["lovable.py", "--list-cases"]):
            from src.benchmarks.lovable_benchmark import main
            main()

    def test_main_export_prompts(self, tmp_path):
        out = tmp_path / "exp.json"
        with patch("sys.argv", ["lovable.py", "--export-prompts", str(out)]):
            from src.benchmarks.lovable_benchmark import main
            main()
            assert out.exists()

    def test_main_eval_specific(self, tmp_path):
        out = tmp_path / "spec.json"
        with patch("sys.argv", ["lovable.py", "--eval", "screenshot_to_code", "--output", str(out)]), \
             patch("src.benchmarks.lovable_benchmark.LovableBenchmark.run_category", return_value={"total_score": 0, "max_score": 0, "percentage": 0, "results": []}):
            from src.benchmarks.lovable_benchmark import main
            main()
            assert out.exists()
