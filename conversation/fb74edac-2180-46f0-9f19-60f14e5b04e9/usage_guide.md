# Global Skills Usage Guide

You have successfully installed a powerful suite of **140+ Skills**, **16 Specialist Agents**, and **11 Workflows** globally. Here is how to use them in Antigravity.

## 1. Using Agents 🕵️‍♂️

Agents are specialized personas. You can invoke them by name in your chat.

**Syntax:** `@<agent-name>` or "Use the <agent-name> agent to..."

**Available Agents (Top Picks):**

- **@backend-specialist**: For Node, Python, Go backend logic.
- **@frontend-specialist**: For React, Vue, CSS work.
- **@security-auditor**: For reviewing code for vulnerabilities.
- **@database-architect**: For schema design and SQL optimization.
- **@devops-engineer**: For CI/CD, Docker, and cloud infra.
- **@project-planner**: For breaking down complex tasks.

**Example:**
> "@security-auditor please review my authentication logic in `auth.py`"

## 2. Using Workflows ⚡

Workflows are automated procedures triggered by slash commands.

**Syntax:** `/<command-name> [arguments]`

**Available Commands:**

- `/brainstorm` - Explore options before coding.
- `/plan` - Create a detailed task breakdown.
- `/debug` - Systematically investigate a bug.
- `/test` - Generate and run tests.
- `/ui-ux-pro-max` - Generate premium designs.

**Example:**
> "/brainstorm ideas for a viral marketing campaign for this app"

## 3. Using Skills 🛠️

Skills are tools that the AI automatically picks up based on context. You generally don't need to "call" them explicitly, but you can hint at them.

**Key Skills & Triggers:**

- **UI/UX Pro Max**: Ask to "Build a landing page" or "Design a dashboard".
- **Pentesting Tools**: Ask to "perform an AWS pentest" or "check for XSS".
- **Framework Guides**: Ask for "React best practices" or "Backend guidelines".
- **Document Tools**: Ask to "Create a PDF" or "Analyze this Excel sheet" (uses `pdf-official`, `xlsx-official`).

**Example:**
> "Design a modern dark-mode dashboard for a fintech app using the UI/UX skill."

## 4. Special Features

- **Design System Generator**: The `/ui-ux-pro-max` command (or just natural language like "Design a SaaS interface") will now generate full color palettes, typography, and layout rules automatically.
- **Validation**: If you ever suspect a skill is broken, you can run the validation script:
  `python3 ~/.agent/.shared/ui-ux-pro-max/scripts/search.py "test" --domain style`

---
*All these features are now globally available in every workspace you open with Antigravity!*
