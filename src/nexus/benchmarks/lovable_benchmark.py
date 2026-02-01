#!/usr/bin/env python3
"""
lovable_benchmark.py

Lovable/Replit-Style Benchmark for UI Code Generation.
Tests the model's ability to generate complete, working UI components from
screenshots, descriptions, and multi-file requirements.

Features:
- Screenshot-to-code generation
- End-to-end feature completion
- Multi-file code generation with dependency resolution
- Component library consistency

Usage:
    python lovable_benchmark.py --eval all
    python lovable_benchmark.py --list-cases
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logging_config import setup_logger
    logger = setup_logger(__name__, "logs/lovable_benchmark.log")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class UIGenCase:
    """Single UI generation evaluation case."""
    id: str
    category: str  # screenshot_to_code, feature_completion, multi_file, component_consistency
    description: str
    prompt: str
    expected_files: List[str]  # Files that should be generated
    required_elements: Dict[str, List[str]]  # file -> required elements
    rubric: Dict[str, int]
    difficulty: str


@dataclass
class UIGenResult:
    """Result of UI generation evaluation."""
    case_id: str
    score: float
    max_score: float
    files_generated: List[str]
    requirements_met: Dict[str, bool]
    response_preview: str


# ═══════════════════════════════════════════════════════════════
# SCREENSHOT TO CODE BENCHMARK
# ═══════════════════════════════════════════════════════════════

SCREENSHOT_TO_CODE_CASES = [
    UIGenCase(
        id="stc_001",
        category="screenshot_to_code",
        description="Login form with email, password, and social login options",
        prompt="""Generate a complete React component for a login form based on this description:
- Clean, modern design with centered card layout
- Email input field with validation
- Password input field with show/hide toggle
- "Remember me" checkbox
- "Forgot password" link
- Primary "Sign In" button
- Social login options (Google, GitHub)
- "Don't have an account? Sign up" link at bottom

Use Tailwind CSS for styling. Include proper form validation and accessibility attributes.""",
        expected_files=["LoginForm.tsx", "LoginForm.css"],
        required_elements={
            "LoginForm.tsx": [
                "email input",
                "password input",
                "show/hide password",
                "remember me",
                "forgot password",
                "submit button",
                "google login",
                "github login",
                "sign up link",
                "form validation",
                "aria- attributes",
            ]
        },
        rubric={
            "correct_structure": 3,
            "all_elements_present": 5,
            "styling_quality": 3,
            "form_validation": 3,
            "accessibility": 3,
            "responsive_design": 2,
            "password_toggle": 2,
        },
        difficulty="medium",
    ),
    UIGenCase(
        id="stc_002",
        category="screenshot_to_code",
        description="Dashboard with sidebar navigation, header, and cards",
        prompt="""Generate a React dashboard layout with:
- Collapsible sidebar with navigation items (Dashboard, Projects, Team, Settings)
- Header with search bar, notifications icon, and user avatar dropdown
- Main content area with 4 stat cards (Total Revenue, Active Users, Orders, Conversion Rate)
- Each card shows an icon, value, percentage change, and sparkline trend

Use Tailwind CSS. Include dark mode support and responsive design.""",
        expected_files=["Dashboard.tsx", "Sidebar.tsx", "Header.tsx", "StatCard.tsx"],
        required_elements={
            "Dashboard.tsx": ["sidebar", "header", "stat cards", "layout grid"],
            "Sidebar.tsx": ["navigation items", "collapse toggle", "icons"],
            "Header.tsx": ["search", "notifications", "user avatar", "dropdown"],
            "StatCard.tsx": ["icon", "value", "change percentage", "trend"],
        },
        rubric={
            "component_structure": 4,
            "all_sections_present": 5,
            "responsive_layout": 3,
            "dark_mode_support": 2,
            "prop_types": 2,
            "styling_consistency": 3,
            "interactive_elements": 2,
        },
        difficulty="hard",
    ),
    UIGenCase(
        id="stc_003",
        category="screenshot_to_code",
        description="E-commerce product card grid",
        prompt="""Generate a React product grid component:
- Grid of 4 product cards per row (responsive: 2 on tablet, 1 on mobile)
- Each card has: image, category tag, product name, star rating, price (with discount), add to cart button
- Hover effect shows quick view button
- Heart icon for wishlist (toggleable)

Include TypeScript interfaces for Product type. Use CSS Grid or Flexbox.""",
        expected_files=["ProductGrid.tsx", "ProductCard.tsx", "types.ts"],
        required_elements={
            "ProductGrid.tsx": ["grid layout", "responsive breakpoints"],
            "ProductCard.tsx": ["image", "category", "name", "rating", "price", "add to cart", "wishlist", "hover effect"],
            "types.ts": ["Product interface", "proper types"],
        },
        rubric={
            "typescript_types": 3,
            "responsive_grid": 3,
            "all_card_elements": 4,
            "hover_interactions": 2,
            "wishlist_toggle": 2,
            "price_formatting": 2,
            "rating_display": 2,
        },
        difficulty="medium",
    ),
]


# ═══════════════════════════════════════════════════════════════
# FEATURE COMPLETION BENCHMARK
# ═══════════════════════════════════════════════════════════════

FEATURE_COMPLETION_CASES = [
    UIGenCase(
        id="fc_001",
        category="feature_completion",
        description="Add infinite scroll to existing blog list",
        prompt="""Given this existing BlogList component, add infinite scroll functionality:

```tsx
function BlogList({ posts }: { posts: Post[] }) {
  return (
    <div className="space-y-4">
      {posts.map(post => (
        <BlogCard key={post.id} post={post} />
      ))}
    </div>
  );
}
```

Requirements:
- Detect when user scrolls to bottom (with 200px threshold)
- Call fetchMorePosts() when threshold reached
- Show loading spinner while fetching
- Handle error state with retry button
- Implement with Intersection Observer API
- Add "No more posts" message when all loaded""",
        expected_files=["BlogList.tsx"],
        required_elements={
            "BlogList.tsx": [
                "IntersectionObserver",
                "loading state",
                "error handling",
                "retry button",
                "no more posts",
                "threshold",
            ]
        },
        rubric={
            "intersection_observer": 4,
            "loading_state": 2,
            "error_handling": 3,
            "retry_logic": 2,
            "end_detection": 2,
            "cleanup": 2,
        },
        difficulty="medium",
    ),
    UIGenCase(
        id="fc_002",
        category="feature_completion",
        description="Add drag-and-drop reordering to todo list",
        prompt="""Modify this TodoList to support drag-and-drop reordering:

```tsx
function TodoList({ todos, setTodos }: Props) {
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>{todo.text}</li>
      ))}
    </ul>
  );
}
```

Requirements:
- Use native HTML5 drag and drop (no external libraries)
- Show visual feedback during drag (opacity change, border)
- Handle drop to reorder items in the list
- Update parent state with new order
- Preserve accessibility (keyboard reorder support)
- Add grip handle icon for drag initiation""",
        expected_files=["TodoList.tsx", "DraggableItem.tsx"],
        required_elements={
            "TodoList.tsx": ["onDragOver", "onDrop", "state update"],
            "DraggableItem.tsx": ["draggable", "onDragStart", "onDragEnd", "grip handle", "visual feedback"],
        },
        rubric={
            "drag_events": 4,
            "visual_feedback": 3,
            "state_update": 3,
            "accessibility": 3,
            "grip_handle": 2,
        },
        difficulty="hard",
    ),
]


# ═══════════════════════════════════════════════════════════════
# MULTI-FILE GENERATION BENCHMARK
# ═══════════════════════════════════════════════════════════════

MULTI_FILE_CASES = [
    UIGenCase(
        id="mf_001",
        category="multi_file",
        description="Complete authentication system with protected routes",
        prompt="""Generate a complete authentication system for a React app:

Files needed:
1. AuthContext.tsx - Context for auth state (user, isAuthenticated, login, logout, loading)
2. useAuth.ts - Hook to access auth context
3. ProtectedRoute.tsx - HOC/component to protect routes
4. LoginPage.tsx - Login page with form
5. api/auth.ts - API functions for login/logout/checkSession

Requirements:
- Store token in localStorage
- Auto-check session on app mount
- Redirect to login if not authenticated
- Redirect to dashboard after login
- Handle loading states
- TypeScript throughout""",
        expected_files=["AuthContext.tsx", "useAuth.ts", "ProtectedRoute.tsx", "LoginPage.tsx", "api/auth.ts"],
        required_elements={
            "AuthContext.tsx": ["createContext", "Provider", "user state", "login", "logout", "loading"],
            "useAuth.ts": ["useContext", "export"],
            "ProtectedRoute.tsx": ["useAuth", "redirect", "loading check"],
            "LoginPage.tsx": ["form", "submit handler", "redirect on success"],
            "api/auth.ts": ["login function", "logout function", "checkSession", "token handling"],
        },
        rubric={
            "all_files_present": 4,
            "context_design": 3,
            "token_handling": 3,
            "protected_route_logic": 3,
            "login_flow": 3,
            "type_safety": 3,
            "error_handling": 2,
        },
        difficulty="hard",
    ),
    UIGenCase(
        id="mf_002",
        category="multi_file",
        description="Complete CRUD for a notes app with optimistic updates",
        prompt="""Generate a complete notes CRUD implementation:

Files needed:
1. NotesPage.tsx - Main page with list and create functionality
2. NoteCard.tsx - Individual note display with edit/delete
3. NoteEditor.tsx - Modal for create/edit note
4. hooks/useNotes.ts - Custom hook for CRUD operations
5. api/notes.ts - API layer
6. types/notes.ts - TypeScript types

Requirements:
- Optimistic updates for create/update/delete
- Rollback on API failure
- Loading states per item
- Toast notifications for success/error
- Confirmation before delete""",
        expected_files=["NotesPage.tsx", "NoteCard.tsx", "NoteEditor.tsx", "hooks/useNotes.ts", "api/notes.ts", "types/notes.ts"],
        required_elements={
            "NotesPage.tsx": ["useNotes hook", "create button", "notes list"],
            "NoteCard.tsx": ["edit button", "delete button", "loading state"],
            "NoteEditor.tsx": ["form", "submit", "cancel"],
            "hooks/useNotes.ts": ["create", "update", "delete", "optimistic update", "rollback"],
            "api/notes.ts": ["CRUD functions", "error handling"],
            "types/notes.ts": ["Note type", "CreateNoteInput"],
        },
        rubric={
            "all_files_present": 3,
            "optimistic_updates": 4,
            "rollback_logic": 3,
            "loading_states": 2,
            "toast_notifications": 2,
            "delete_confirmation": 2,
            "type_definitions": 2,
        },
        difficulty="hard",
    ),
]


# ═══════════════════════════════════════════════════════════════
# COMPONENT CONSISTENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════

COMPONENT_CONSISTENCY_CASES = [
    UIGenCase(
        id="cc_001",
        category="component_consistency",
        description="Generate consistent button variants",
        prompt="""Create a Button component library with consistent design tokens:

Variants needed:
- Primary (filled, brand color)
- Secondary (outlined)
- Ghost (transparent background)
- Danger (destructive actions)

Sizes: sm, md, lg

States: default, hover, active, disabled, loading

Requirements:
- Consistent spacing scale
- Accessible color contrast
- Focus visible states
- Loading spinner replaces text
- TypeScript with proper props
- Support for icons (left/right)
- asChild pattern for composition""",
        expected_files=["Button.tsx", "Button.stories.tsx", "button.css"],
        required_elements={
            "Button.tsx": [
                "variants prop",
                "size prop",
                "disabled state",
                "loading state",
                "icon support",
                "asChild",
                "forwardRef",
            ],
            "Button.stories.tsx": ["all variants", "all sizes", "loading story"],
        },
        rubric={
            "variant_system": 4,
            "size_system": 2,
            "loading_state": 2,
            "icon_support": 2,
            "typescript_props": 3,
            "accessibility": 3,
            "forward_ref": 2,
        },
        difficulty="medium",
    ),
]


# ═══════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════

class LovableBenchmark:
    """Main benchmark runner for Lovable-style UI generation."""
    
    CASES = {
        "screenshot_to_code": SCREENSHOT_TO_CODE_CASES,
        "feature_completion": FEATURE_COMPLETION_CASES,
        "multi_file": MULTI_FILE_CASES,
        "component_consistency": COMPONENT_CONSISTENCY_CASES,
    }
    
    def __init__(self, model_fn=None):
        """
        Initialize benchmark runner.
        
        Args:
            model_fn: Function that takes a prompt and returns a response.
        """
        self.model_fn = model_fn or self._dummy_model
        self.results = []
    
    def _dummy_model(self, prompt: str) -> str:
        """Dummy model for testing."""
        return f"// Generated code for: {prompt[:100]}..."
    
    def evaluate_response(self, case: UIGenCase, response: str) -> UIGenResult:
        """Evaluate a model response for a UI generation case."""
        response_lower = response.lower()
        
        # Check which files appear to be generated
        files_generated = []
        for expected_file in case.expected_files:
            file_base = expected_file.split('.')[0].lower()
            if file_base in response_lower or expected_file.lower() in response_lower:
                files_generated.append(expected_file)
        
        # Check requirements per file
        requirements_met = {}
        for file_name, requirements in case.required_elements.items():
            file_base = file_name.split('.')[0].lower()
            for req in requirements:
                req_key = f"{file_name}:{req}"
                # Simple keyword check
                requirements_met[req_key] = req.lower() in response_lower or any(
                    word in response_lower for word in req.lower().split()
                )
        
        # Score based on rubric
        score = 0
        for criterion, points in case.rubric.items():
            criterion_lower = criterion.lower().replace("_", " ")
            # Heuristic: check if relevant keywords appear
            criterion_passed = False
            
            # Map rubric criteria to expected patterns
            if "files" in criterion_lower or "present" in criterion_lower:
                criterion_passed = len(files_generated) >= len(case.expected_files) * 0.7
            elif "typescript" in criterion_lower or "type" in criterion_lower:
                criterion_passed = any(t in response_lower for t in ["interface", "type ", ": string", ": number", ": boolean"])
            elif "loading" in criterion_lower:
                criterion_passed = any(l in response_lower for l in ["loading", "isloading", "spinner"])
            elif "error" in criterion_lower:
                criterion_passed = any(e in response_lower for e in ["error", "catch", "try"])
            elif "accessibility" in criterion_lower or "a11y" in criterion_lower:
                criterion_passed = any(a in response_lower for a in ["aria-", "role=", "tabindex"])
            elif "responsive" in criterion_lower:
                criterion_passed = any(r in response_lower for r in ["md:", "lg:", "sm:", "@media", "grid-cols"])
            elif "variant" in criterion_lower:
                criterion_passed = "variant" in response_lower
            elif "optimistic" in criterion_lower:
                criterion_passed = any(o in response_lower for o in ["optimistic", "rollback", "pending"])
            else:
                # Generic: check if response is substantial
                criterion_passed = len(response) > 500
            
            if criterion_passed:
                score += points
        
        return UIGenResult(
            case_id=case.id,
            score=score,
            max_score=sum(case.rubric.values()),
            files_generated=files_generated,
            requirements_met=requirements_met,
            response_preview=response[:500],
        )
    
    def run_category(self, category: str) -> Dict[str, Any]:
        """Run evaluation for a specific category."""
        if category not in self.CASES:
            raise ValueError(f"Unknown category: {category}")
        
        cases = self.CASES[category]
        results = []
        total_score = 0
        max_score = 0
        
        for case in cases:
            logger.info(f"Running case: {case.id}")
            response = self.model_fn(case.prompt)
            result = self.evaluate_response(case, response)
            results.append(asdict(result))
            total_score += result.score
            max_score += result.max_score
        
        return {
            "category": category,
            "cases": len(cases),
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(total_score / max_score * 100, 2) if max_score > 0 else 0,
            "results": results,
        }
    
    def run_all(self) -> Dict[str, Any]:
        """Run all evaluation categories."""
        all_results = {}
        total_score = 0
        max_score = 0
        
        for category in self.CASES:
            logger.info(f"\n{'='*60}\nCategory: {category}\n{'='*60}")
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
        """Save results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    def get_all_cases(self) -> List[UIGenCase]:
        """Get all evaluation cases."""
        all_cases = []
        for cases in self.CASES.values():
            all_cases.extend(cases)
        return all_cases
    
    def export_prompts(self, output_path: Path):
        """Export all prompts for manual evaluation."""
        prompts = []
        for category, cases in self.CASES.items():
            for case in cases:
                prompts.append({
                    "id": case.id,
                    "category": category,
                    "description": case.description,
                    "prompt": case.prompt,
                    "expected_files": case.expected_files,
                    "difficulty": case.difficulty,
                    "max_score": sum(case.rubric.values()),
                })
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        logger.info(f"Prompts exported to {output_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Lovable-Style UI Generation Benchmark")
    parser.add_argument(
        "--eval",
        type=str,
        default="all",
        help="Categories to evaluate (comma-separated or 'all')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/lovable_benchmark_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all evaluation cases and exit",
    )
    parser.add_argument(
        "--export-prompts",
        type=str,
        default=None,
        help="Export prompts to JSON file for manual evaluation",
    )
    args = parser.parse_args()
    
    benchmark = LovableBenchmark()
    
    if args.list_cases:
        cases = benchmark.get_all_cases()
        print(f"\nTotal cases: {len(cases)}\n")
        for case in cases:
            print(f"[{case.category}] {case.id} ({case.difficulty})")
            print(f"  Description: {case.description}")
            print(f"  Expected files: {case.expected_files}")
            print(f"  Max score: {sum(case.rubric.values())}\n")
        return
    
    if args.export_prompts:
        benchmark.export_prompts(Path(args.export_prompts))
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
    print("LOVABLE BENCHMARK RESULTS")
    print("=" * 60)
    
    if "overall_percentage" in results:
        print(f"\nOverall Score: {results['overall_score']}/{results['overall_max']} ({results['overall_percentage']}%)")
    
    if "categories" in results:
        print("\nCategory Breakdown:")
        for cat, cat_result in results["categories"].items():
            print(f"  {cat}: {cat_result['total_score']}/{cat_result['max_score']} ({cat_result['percentage']}%)")


if __name__ == "__main__":
    main()
