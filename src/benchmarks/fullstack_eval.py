#!/usr/bin/env python3
"""
fullstack_eval.py

FullstackEval-2025: Benchmark suite for evaluating fullstack engineering capabilities.
Tests API design, database queries, frontend components, DevOps, testing, and more.

Evaluation Categories:
- REST API Design Correctness
- SQL Schema Optimization  
- React Component Accessibility
- Kubernetes Manifest Validation
- Terraform IaC Best Practices
- CI/CD Pipeline Design

Usage:
    python fullstack_eval.py --model gpt-4 --eval all
    python fullstack_eval.py --model local --eval api,sql
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logging_config import setup_logger
    logger = setup_logger(__name__, "logs/fullstack_eval.log")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvalCase:
    """Single evaluation case."""
    id: str
    category: str
    prompt: str
    expected_elements: List[str]  # Key elements that should be in response
    rubric: Dict[str, int]  # Scoring rubric (criteria -> max points)
    difficulty: str  # easy, medium, hard


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    case_id: str
    score: float
    max_score: float
    passed_criteria: List[str]
    failed_criteria: List[str]
    response: str


# ═══════════════════════════════════════════════════════════════
# BASE EVALUATOR
# ═══════════════════════════════════════════════════════════════

class BaseEvaluator(ABC):
    """Base class for evaluators."""
    
    @abstractmethod
    def get_cases(self) -> List[EvalCase]:
        """Return list of evaluation cases."""
        pass
    
    @abstractmethod
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        """Evaluate a response for a given case."""
        pass


# ═══════════════════════════════════════════════════════════════
# REST API EVALUATOR
# ═══════════════════════════════════════════════════════════════

class RESTApiEvaluator(BaseEvaluator):
    """Evaluates REST API design quality."""
    
    def get_cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id="api_001",
                category="rest_api",
                prompt="Design a RESTful API endpoint for creating a new user account. Include the endpoint, HTTP method, request body schema, and response format for both success and error cases.",
                expected_elements=[
                    "POST method",
                    "users endpoint",
                    "email validation",
                    "password requirements",
                    "201 created",
                    "400 or 422 for validation",
                    "409 for duplicate",
                ],
                rubric={
                    "correct_http_method": 2,
                    "proper_endpoint_naming": 2,
                    "input_validation": 3,
                    "success_response": 2,
                    "error_handling": 3,
                    "status_codes": 2,
                    "security_considerations": 2,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="api_002",
                category="rest_api",
                prompt="Design a paginated API endpoint for listing products with sorting and filtering capabilities. The endpoint should support filtering by category, price range, and search term.",
                expected_elements=[
                    "GET method",
                    "query parameters",
                    "pagination (limit/offset or cursor)",
                    "sorting parameter",
                    "filter parameters",
                    "total count in response",
                ],
                rubric={
                    "correct_http_method": 1,
                    "pagination_design": 3,
                    "filter_parameters": 3,
                    "sorting_support": 2,
                    "response_structure": 2,
                    "performance_considerations": 2,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="api_003",
                category="rest_api",
                prompt="Design an API for a shopping cart with operations: add item, update quantity, remove item, and checkout. Include proper error handling and concurrent access considerations.",
                expected_elements=[
                    "cart resource",
                    "items subresource",
                    "atomic operations",
                    "concurrency handling",
                    "checkout as action",
                ],
                rubric={
                    "resource_modeling": 3,
                    "crud_operations": 2,
                    "checkout_design": 2,
                    "concurrency_handling": 3,
                    "error_scenarios": 2,
                    "idempotency": 2,
                },
                difficulty="hard",
            ),
        ]
    
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        response_lower = response.lower()
        passed = []
        failed = []
        score = 0
        
        for criterion, points in case.rubric.items():
            # Simple keyword-based evaluation (would be LLM-graded in production)
            criterion_passed = False
            
            if criterion == "correct_http_method":
                criterion_passed = any(m in response_lower for m in ["post", "get", "put", "patch", "delete"])
            elif criterion == "proper_endpoint_naming":
                criterion_passed = "/" in response and any(r in response_lower for r in ["users", "products", "cart"])
            elif criterion == "input_validation":
                criterion_passed = any(v in response_lower for v in ["validation", "validate", "required", "schema"])
            elif criterion == "success_response":
                criterion_passed = any(s in response_lower for s in ["200", "201", "success", "created"])
            elif criterion == "error_handling":
                criterion_passed = any(e in response_lower for e in ["error", "400", "404", "422", "500"])
            elif criterion == "status_codes":
                criterion_passed = any(str(c) in response for c in [200, 201, 204, 400, 401, 403, 404, 422, 500])
            elif criterion == "security_considerations":
                criterion_passed = any(s in response_lower for s in ["auth", "token", "jwt", "password", "hash"])
            elif criterion == "pagination_design":
                criterion_passed = any(p in response_lower for p in ["page", "limit", "offset", "cursor", "next"])
            elif criterion == "filter_parameters":
                criterion_passed = any(f in response_lower for f in ["filter", "category", "price", "search", "query"])
            elif criterion == "sorting_support":
                criterion_passed = any(s in response_lower for s in ["sort", "order", "asc", "desc"])
            elif criterion == "response_structure":
                criterion_passed = any(r in response_lower for r in ["data", "items", "total", "meta"])
            elif criterion == "performance_considerations":
                criterion_passed = any(p in response_lower for p in ["index", "cache", "limit", "efficient"])
            elif criterion == "resource_modeling":
                criterion_passed = "/" in response and "cart" in response_lower
            elif criterion == "crud_operations":
                criterion_passed = sum(1 for m in ["post", "get", "put", "delete", "patch"] if m in response_lower) >= 2
            elif criterion == "checkout_design":
                criterion_passed = "checkout" in response_lower
            elif criterion == "concurrency_handling":
                criterion_passed = any(c in response_lower for c in ["concurrent", "lock", "optimistic", "version", "etag"])
            elif criterion == "idempotency":
                criterion_passed = any(i in response_lower for i in ["idempoten", "idempotency", "retry", "key"])
            else:
                criterion_passed = len(response) > 100
            
            if criterion_passed:
                passed.append(criterion)
                score += points
            else:
                failed.append(criterion)
        
        return EvalResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            passed_criteria=passed,
            failed_criteria=failed,
            response=response[:500],
        )


# ═══════════════════════════════════════════════════════════════
# SQL EVALUATOR
# ═══════════════════════════════════════════════════════════════

class SQLEvaluator(BaseEvaluator):
    """Evaluates SQL schema and query design."""
    
    def get_cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id="sql_001",
                category="sql",
                prompt="Design a database schema for an e-commerce system with users, products, orders, and order items. Include primary keys, foreign keys, and indexes.",
                expected_elements=[
                    "users table",
                    "products table", 
                    "orders table",
                    "order_items table",
                    "foreign keys",
                    "indexes",
                ],
                rubric={
                    "table_design": 3,
                    "primary_keys": 2,
                    "foreign_keys": 3,
                    "indexes": 2,
                    "data_types": 2,
                    "normalization": 2,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="sql_002",
                category="sql",
                prompt="Write an optimized SQL query to find the top 10 customers by total order value in the last 30 days, including their email and number of orders.",
                expected_elements=[
                    "join",
                    "sum",
                    "group by",
                    "order by",
                    "limit 10",
                    "date filter",
                ],
                rubric={
                    "correct_joins": 3,
                    "aggregation": 2,
                    "grouping": 2,
                    "ordering": 1,
                    "date_handling": 2,
                    "optimization": 2,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="sql_003",
                category="sql",
                prompt="Write a migration script to add a 'status' column to an orders table with 1 million rows without causing downtime.",
                expected_elements=[
                    "add column",
                    "nullable or default",
                    "backfill",
                    "not null constraint",
                    "batches",
                ],
                rubric={
                    "safe_column_add": 3,
                    "backfill_strategy": 3,
                    "constraint_handling": 2,
                    "rollback_plan": 2,
                    "no_downtime": 3,
                },
                difficulty="hard",
            ),
        ]
    
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        response_lower = response.lower()
        passed = []
        failed = []
        score = 0
        
        for criterion, points in case.rubric.items():
            criterion_passed = False
            
            if criterion == "table_design":
                criterion_passed = sum(1 for t in ["create table", "table", "users", "products", "orders"] if t in response_lower) >= 2
            elif criterion == "primary_keys":
                criterion_passed = "primary key" in response_lower or "pk" in response_lower
            elif criterion == "foreign_keys":
                criterion_passed = "foreign key" in response_lower or "references" in response_lower or "fk" in response_lower
            elif criterion == "indexes":
                criterion_passed = "index" in response_lower or "create index" in response_lower
            elif criterion == "data_types":
                criterion_passed = any(d in response_lower for d in ["varchar", "int", "decimal", "timestamp", "text", "boolean"])
            elif criterion == "normalization":
                criterion_passed = len([t for t in ["users", "products", "orders", "order_items"] if t in response_lower]) >= 3
            elif criterion == "correct_joins":
                criterion_passed = "join" in response_lower
            elif criterion == "aggregation":
                criterion_passed = any(a in response_lower for a in ["sum", "count", "avg", "max", "min"])
            elif criterion == "grouping":
                criterion_passed = "group by" in response_lower
            elif criterion == "ordering":
                criterion_passed = "order by" in response_lower
            elif criterion == "date_handling":
                criterion_passed = any(d in response_lower for d in ["30 day", "interval", "date", "now()", "current_date"])
            elif criterion == "optimization":
                criterion_passed = any(o in response_lower for o in ["index", "limit", "explain", "efficient"])
            elif criterion == "safe_column_add":
                criterion_passed = "add column" in response_lower and ("null" in response_lower or "default" in response_lower)
            elif criterion == "backfill_strategy":
                criterion_passed = any(b in response_lower for b in ["batch", "update", "where", "chunk"])
            elif criterion == "constraint_handling":
                criterion_passed = any(c in response_lower for c in ["not null", "constraint", "alter"])
            elif criterion == "rollback_plan":
                criterion_passed = any(r in response_lower for r in ["rollback", "drop", "revert", "if fail"])
            elif criterion == "no_downtime":
                criterion_passed = any(n in response_lower for n in ["no lock", "online", "concurrent", "batch"])
            else:
                criterion_passed = len(response) > 50
            
            if criterion_passed:
                passed.append(criterion)
                score += points
            else:
                failed.append(criterion)
        
        return EvalResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            passed_criteria=passed,
            failed_criteria=failed,
            response=response[:500],
        )


# ═══════════════════════════════════════════════════════════════
# REACT ACCESSIBILITY EVALUATOR
# ═══════════════════════════════════════════════════════════════

class ReactAccessibilityEvaluator(BaseEvaluator):
    """Evaluates React component accessibility."""
    
    def get_cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id="react_001",
                category="react_a11y",
                prompt="Create an accessible modal dialog component in React with proper keyboard navigation and screen reader support.",
                expected_elements=[
                    "role=dialog",
                    "aria-modal",
                    "aria-labelledby",
                    "focus trap",
                    "escape to close",
                    "focus return",
                ],
                rubric={
                    "aria_attributes": 3,
                    "keyboard_navigation": 3,
                    "focus_management": 3,
                    "semantic_markup": 2,
                    "proper_props": 2,
                },
                difficulty="hard",
            ),
            EvalCase(
                id="react_002",
                category="react_a11y",
                prompt="Create an accessible form with labels, error messages, and required field indicators.",
                expected_elements=[
                    "label htmlFor",
                    "aria-describedby",
                    "aria-invalid",
                    "aria-required",
                    "error association",
                ],
                rubric={
                    "label_association": 3,
                    "error_messages": 2,
                    "required_indicators": 2,
                    "aria_attributes": 2,
                    "form_structure": 2,
                },
                difficulty="medium",
            ),
        ]
    
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        response_lower = response.lower()
        passed = []
        failed = []
        score = 0
        
        for criterion, points in case.rubric.items():
            criterion_passed = False
            
            if criterion == "aria_attributes":
                criterion_passed = sum(1 for a in ["aria-", "role="] if a in response_lower) >= 2
            elif criterion == "keyboard_navigation":
                criterion_passed = any(k in response_lower for k in ["onkeydown", "escape", "enter", "tab", "keycode"])
            elif criterion == "focus_management":
                criterion_passed = any(f in response_lower for f in ["focus", "ref", "tabindex", "autofocus"])
            elif criterion == "semantic_markup":
                criterion_passed = any(s in response_lower for s in ["<dialog", "<form", "<button", "<label", "semantic"])
            elif criterion == "proper_props":
                criterion_passed = "props" in response_lower or "interface" in response_lower
            elif criterion == "label_association":
                criterion_passed = "htmlfor" in response_lower or "for=" in response_lower or "id=" in response_lower
            elif criterion == "error_messages":
                criterion_passed = any(e in response_lower for e in ["error", "invalid", "aria-describedby"])
            elif criterion == "required_indicators":
                criterion_passed = any(r in response_lower for r in ["required", "aria-required", "*"])
            elif criterion == "form_structure":
                criterion_passed = "<form" in response_lower or "onsubmit" in response_lower
            else:
                criterion_passed = len(response) > 100
            
            if criterion_passed:
                passed.append(criterion)
                score += points
            else:
                failed.append(criterion)
        
        return EvalResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            passed_criteria=passed,
            failed_criteria=failed,
            response=response[:500],
        )


# ═══════════════════════════════════════════════════════════════
# KUBERNETES EVALUATOR
# ═══════════════════════════════════════════════════════════════

class KubernetesEvaluator(BaseEvaluator):
    """Evaluates Kubernetes manifest design."""
    
    def get_cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id="k8s_001",
                category="kubernetes",
                prompt="Create a Kubernetes deployment for a web application with proper resource limits, health checks, and horizontal pod autoscaling.",
                expected_elements=[
                    "Deployment",
                    "resources limits",
                    "livenessProbe",
                    "readinessProbe",
                    "HPA",
                    "replicas",
                ],
                rubric={
                    "deployment_structure": 2,
                    "resource_limits": 3,
                    "health_probes": 3,
                    "hpa_config": 3,
                    "security_context": 2,
                    "labels_annotations": 1,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="k8s_002",
                category="kubernetes",
                prompt="Design a Kubernetes configuration for a stateful database with persistent storage, secrets management, and backup considerations.",
                expected_elements=[
                    "StatefulSet",
                    "PersistentVolumeClaim",
                    "Secret",
                    "volumeMounts",
                    "storage class",
                ],
                rubric={
                    "statefulset_design": 3,
                    "persistent_storage": 3,
                    "secrets_handling": 2,
                    "volume_config": 2,
                    "backup_strategy": 2,
                },
                difficulty="hard",
            ),
        ]
    
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        response_lower = response.lower()
        passed = []
        failed = []
        score = 0
        
        for criterion, points in case.rubric.items():
            criterion_passed = False
            
            if criterion == "deployment_structure":
                criterion_passed = "kind: deployment" in response_lower or "deployment" in response_lower
            elif criterion == "resource_limits":
                criterion_passed = "resources:" in response_lower and any(r in response_lower for r in ["limits:", "requests:", "cpu:", "memory:"])
            elif criterion == "health_probes":
                criterion_passed = "livenessprobe" in response_lower or "readinessprobe" in response_lower
            elif criterion == "hpa_config":
                criterion_passed = "horizontalpodautoscaler" in response_lower or "hpa" in response_lower or "autoscal" in response_lower
            elif criterion == "security_context":
                criterion_passed = "securitycontext" in response_lower or "runasnonroot" in response_lower
            elif criterion == "labels_annotations":
                criterion_passed = "labels:" in response_lower or "annotations:" in response_lower
            elif criterion == "statefulset_design":
                criterion_passed = "statefulset" in response_lower
            elif criterion == "persistent_storage":
                criterion_passed = "persistentvolumeclaim" in response_lower or "pvc" in response_lower
            elif criterion == "secrets_handling":
                criterion_passed = "secret" in response_lower and ("envfrom" in response_lower or "secretref" in response_lower or "volumemount" in response_lower)
            elif criterion == "volume_config":
                criterion_passed = "volumes:" in response_lower or "volumemounts:" in response_lower
            elif criterion == "backup_strategy":
                criterion_passed = any(b in response_lower for b in ["backup", "snapshot", "velero", "dump"])
            else:
                criterion_passed = len(response) > 100
            
            if criterion_passed:
                passed.append(criterion)
                score += points
            else:
                failed.append(criterion)
        
        return EvalResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            passed_criteria=passed,
            failed_criteria=failed,
            response=response[:500],
        )


# ═══════════════════════════════════════════════════════════════
# TERRAFORM EVALUATOR
# ═══════════════════════════════════════════════════════════════

class TerraformEvaluator(BaseEvaluator):
    """Evaluates Terraform IaC best practices."""
    
    def get_cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id="tf_001",
                category="terraform",
                prompt="Write Terraform code to provision an AWS VPC with public and private subnets, NAT gateway, and proper routing tables.",
                expected_elements=[
                    "aws_vpc",
                    "aws_subnet",
                    "aws_nat_gateway",
                    "aws_route_table",
                    "cidr_block",
                ],
                rubric={
                    "vpc_resource": 2,
                    "subnet_design": 3,
                    "nat_gateway": 2,
                    "routing": 2,
                    "variables": 2,
                    "tags": 1,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="tf_002",
                category="terraform",
                prompt="Create a Terraform module for deploying a highly available RDS database with encryption, backups, and monitoring.",
                expected_elements=[
                    "aws_db_instance",
                    "multi_az",
                    "encrypted",
                    "backup_retention",
                    "module",
                ],
                rubric={
                    "module_structure": 2,
                    "rds_config": 2,
                    "high_availability": 2,
                    "encryption": 2,
                    "backups": 2,
                    "monitoring": 2,
                },
                difficulty="hard",
            ),
        ]
    
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        response_lower = response.lower()
        passed = []
        failed = []
        score = 0
        
        for criterion, points in case.rubric.items():
            criterion_passed = False
            
            if criterion == "vpc_resource":
                criterion_passed = "aws_vpc" in response_lower or "resource \"aws_vpc\"" in response_lower
            elif criterion == "subnet_design":
                criterion_passed = "aws_subnet" in response_lower and ("public" in response_lower or "private" in response_lower)
            elif criterion == "nat_gateway":
                criterion_passed = "nat_gateway" in response_lower or "aws_nat_gateway" in response_lower
            elif criterion == "routing":
                criterion_passed = "route_table" in response_lower or "aws_route" in response_lower
            elif criterion == "variables":
                criterion_passed = "variable" in response_lower or "var." in response_lower
            elif criterion == "tags":
                criterion_passed = "tags" in response_lower
            elif criterion == "module_structure":
                criterion_passed = "module" in response_lower or "output" in response_lower
            elif criterion == "rds_config":
                criterion_passed = "aws_db_instance" in response_lower or "rds" in response_lower
            elif criterion == "high_availability":
                criterion_passed = "multi_az" in response_lower or "multi-az" in response_lower
            elif criterion == "encryption":
                criterion_passed = "encrypt" in response_lower or "kms" in response_lower
            elif criterion == "backups":
                criterion_passed = "backup" in response_lower or "retention" in response_lower
            elif criterion == "monitoring":
                criterion_passed = any(m in response_lower for m in ["cloudwatch", "monitoring", "alarm", "enhanced_monitoring"])
            else:
                criterion_passed = len(response) > 100
            
            if criterion_passed:
                passed.append(criterion)
                score += points
            else:
                failed.append(criterion)
        
        return EvalResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            passed_criteria=passed,
            failed_criteria=failed,
            response=response[:500],
        )


# ═══════════════════════════════════════════════════════════════
# CI/CD EVALUATOR
# ═══════════════════════════════════════════════════════════════

class CICDEvaluator(BaseEvaluator):
    """Evaluates CI/CD pipeline design."""
    
    def get_cases(self) -> List[EvalCase]:
        return [
            EvalCase(
                id="cicd_001",
                category="cicd",
                prompt="Design a GitHub Actions workflow for a Node.js application that runs tests, builds, and deploys to production with proper caching and secrets management.",
                expected_elements=[
                    "actions/checkout",
                    "actions/setup-node",
                    "npm test",
                    "npm run build",
                    "cache",
                    "secrets",
                ],
                rubric={
                    "workflow_structure": 2,
                    "build_steps": 2,
                    "test_steps": 2,
                    "caching": 2,
                    "secrets_usage": 2,
                    "deployment": 2,
                },
                difficulty="medium",
            ),
            EvalCase(
                id="cicd_002",
                category="cicd",
                prompt="Create a multi-stage deployment pipeline with staging and production environments, including approval gates and rollback capabilities.",
                expected_elements=[
                    "staging",
                    "production",
                    "approval",
                    "rollback",
                    "environment",
                ],
                rubric={
                    "multi_env": 3,
                    "approval_gates": 3,
                    "rollback_strategy": 3,
                    "environment_config": 2,
                    "deployment_strategy": 2,
                },
                difficulty="hard",
            ),
        ]
    
    def evaluate_response(self, case: EvalCase, response: str) -> EvalResult:
        response_lower = response.lower()
        passed = []
        failed = []
        score = 0
        
        for criterion, points in case.rubric.items():
            criterion_passed = False
            
            if criterion == "workflow_structure":
                criterion_passed = "name:" in response_lower and ("on:" in response_lower or "jobs:" in response_lower)
            elif criterion == "build_steps":
                criterion_passed = any(b in response_lower for b in ["npm run build", "npm build", "build:", "run"])
            elif criterion == "test_steps":
                criterion_passed = any(t in response_lower for t in ["npm test", "npm run test", "test:", "jest", "pytest"])
            elif criterion == "caching":
                criterion_passed = "cache" in response_lower or "actions/cache" in response_lower
            elif criterion == "secrets_usage":
                criterion_passed = "secrets." in response_lower or "${{ secrets" in response_lower
            elif criterion == "deployment":
                criterion_passed = any(d in response_lower for d in ["deploy", "push", "upload", "release"])
            elif criterion == "multi_env":
                criterion_passed = "staging" in response_lower and "production" in response_lower
            elif criterion == "approval_gates":
                criterion_passed = any(a in response_lower for a in ["approval", "manual", "review", "confirm", "gate"])
            elif criterion == "rollback_strategy":
                criterion_passed = any(r in response_lower for r in ["rollback", "revert", "previous", "fail"])
            elif criterion == "environment_config":
                criterion_passed = "environment:" in response_lower or "env:" in response_lower
            elif criterion == "deployment_strategy":
                criterion_passed = any(s in response_lower for s in ["canary", "blue-green", "rolling", "progressive"])
            else:
                criterion_passed = len(response) > 100
            
            if criterion_passed:
                passed.append(criterion)
                score += points
            else:
                failed.append(criterion)
        
        return EvalResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            passed_criteria=passed,
            failed_criteria=failed,
            response=response[:500],
        )


# ═══════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════

class FullstackEval:
    """Main benchmark runner for fullstack evaluation."""
    
    EVALUATORS = {
        "api": RESTApiEvaluator(),
        "sql": SQLEvaluator(),
        "react": ReactAccessibilityEvaluator(),
        "k8s": KubernetesEvaluator(),
        "terraform": TerraformEvaluator(),
        "cicd": CICDEvaluator(),
    }
    
    def __init__(self, model_fn=None):
        """
        Initialize benchmark runner.
        
        Args:
            model_fn: Function that takes a prompt and returns a response.
                      If None, uses a dummy response for testing.
        """
        self.model_fn = model_fn or self._dummy_model
        self.results = []
    
    def _dummy_model(self, prompt: str) -> str:
        """Dummy model for testing evaluation logic."""
        return f"Response to: {prompt[:50]}..."
    
    def run_category(self, category: str) -> Dict[str, Any]:
        """Run evaluation for a specific category."""
        if category not in self.EVALUATORS:
            raise ValueError(f"Unknown category: {category}. Available: {list(self.EVALUATORS.keys())}")
        
        evaluator = self.EVALUATORS[category]
        cases = evaluator.get_cases()
        
        category_results = []
        total_score = 0
        max_score = 0
        
        for case in cases:
            logger.info(f"Running case: {case.id}")
            response = self.model_fn(case.prompt)
            result = evaluator.evaluate_response(case, response)
            category_results.append(asdict(result))
            total_score += result.score
            max_score += result.max_score
        
        return {
            "category": category,
            "cases": len(cases),
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(total_score / max_score * 100, 2) if max_score > 0 else 0,
            "results": category_results,
        }
    
    def run_all(self) -> Dict[str, Any]:
        """Run all evaluation categories."""
        all_results = {}
        total_score = 0
        max_score = 0
        
        for category in self.EVALUATORS:
            logger.info(f"\n{'='*60}\nRunning category: {category}\n{'='*60}")
            result = self.run_category(category)
            all_results[category] = result
            total_score += result["total_score"]
            max_score += result["max_score"]
        
        return {
            "overall_score": total_score,
            "overall_max": max_score,
            "overall_percentage": round(total_score / max_score * 100, 2) if max_score > 0 else 0,
            "categories": all_results,
        }
    
    def save_results(self, output_path: Path, results: Dict):
        """Save evaluation results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    def get_all_cases(self) -> List[EvalCase]:
        """Get all evaluation cases across all categories."""
        all_cases = []
        for evaluator in self.EVALUATORS.values():
            all_cases.extend(evaluator.get_cases())
        return all_cases


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FullstackEval-2025 Benchmark Suite")
    parser.add_argument(
        "--eval",
        type=str,
        default="all",
        help="Categories to evaluate (comma-separated or 'all')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/fullstack_eval_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all evaluation cases and exit",
    )
    args = parser.parse_args()
    
    benchmark = FullstackEval()
    
    if args.list_cases:
        cases = benchmark.get_all_cases()
        print(f"\nTotal cases: {len(cases)}\n")
        for case in cases:
            print(f"[{case.category}] {case.id} ({case.difficulty})")
            print(f"  Prompt: {case.prompt[:80]}...")
            print(f"  Criteria: {list(case.rubric.keys())}\n")
        return
    
    if args.eval == "all":
        results = benchmark.run_all()
    else:
        categories = args.eval.split(",")
        results = {"categories": {}}
        for cat in categories:
            results["categories"][cat.strip()] = benchmark.run_category(cat.strip())
    
    benchmark.save_results(Path(args.output), results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FULLSTACK EVAL RESULTS")
    print("=" * 60)
    
    if "overall_percentage" in results:
        print(f"\nOverall Score: {results['overall_score']}/{results['overall_max']} ({results['overall_percentage']}%)")
    
    if "categories" in results:
        print("\nCategory Breakdown:")
        for cat, cat_result in results["categories"].items():
            print(f"  {cat}: {cat_result['total_score']}/{cat_result['max_score']} ({cat_result['percentage']}%)")


if __name__ == "__main__":
    main()
