# AGENTS.md

## Project purpose

This repository is an educational machine learning project about Responsible Gaming.

The goal is to learn and understand an end-to-end binary classification workflow using entirely synthetic player data.

The model predicts one of two educational labels:

- RISK
- NO_RISK

This project must never be presented as a real diagnosis, clinical assessment, or production Responsible Gaming decision system.

## Communication

- Communicate with the user in French.
- Write code, identifiers, filenames, comments, and technical documentation in English.
- Explain unfamiliar ML concepts before implementing them.
- Prefer simple explanations before introducing mathematical or technical details.
- If an implementation introduces a concept the user has not encountered yet, explain it before proceeding.
- Do not generate code the user cannot reasonably understand without first explaining the relevant concept.

## Development principles

- Prefer simple, readable, explicit code.
- Keep functions small and focused.
- Use type hints when they improve clarity.
- Use descriptive names.
- Avoid premature abstractions.
- Avoid over-engineering.
- Do not introduce architecture patterns unless the current problem justifies them.
- Do not refactor unrelated code.
- Do not implement future roadmap steps unless explicitly requested.
- Prefer the smallest implementation that correctly solves the current task.

The goal is not to build the project as fast as possible.

The goal is to keep every implementation understandable, testable, and reviewable.

## Machine learning principles

Do not hide important ML concepts behind unnecessary abstractions.

When relevant, keep the following concepts visible and understandable:

- features and target
- training data
- validation data
- test data
- preprocessing
- baseline models
- model training
- predictions
- evaluation metrics
- confusion matrix
- false positives and false negatives
- class imbalance
- overfitting and underfitting
- regularization
- hyperparameters

Prefer pedagogical clarity over model complexity or maximum performance.

When introducing a new ML concept:

1. Explain what it is.
2. Explain why it is needed in the current task.
3. Show where it appears in the implementation.
4. Run or observe the result.
5. Explain what the result means.

Do not introduce advanced models when a simpler model is sufficient to teach the current concept.

## Data and safety

- Use only synthetic or fictitious player data.
- Never introduce real player data or personal data.
- Never present model output as a diagnosis of gambling addiction or problem gambling.
- Never claim that the educational model is suitable for real-world Responsible Gaming decisions.
- Never commit credentials, API keys, AWS secrets, tokens, or sensitive `.env` files.

## Tooling

The project currently uses:

- Python 3.13
- uv
- pandas
- NumPy
- scikit-learn
- matplotlib
- pytest
- Ruff

Use `uv` for Python dependency management and command execution.

Do not use `pip` directly unless explicitly requested.

Do not introduce new dependencies without explaining why they are needed.

PyTorch and AWS/SageMaker are outside the current scope unless explicitly requested.

## Project structure

Keep the repository structure simple.

Current high-level structure:

```text
responsible-gaming-ml/
├── data/
├── src/
│   └── responsible_gaming_ml/
├── tests/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

Do not create directories, modules, layers, or abstractions for hypothetical future requirements.

Let the structure evolve when concrete requirements justify it.

## Testing principles

- Add tests when they protect meaningful behavior.
- Prefer small and explicit tests.
- Tests should be easy to read and understand.
- Do not create tests only to increase test count or coverage.
- Test important data transformations and application behavior when relevant.
- Keep tests deterministic.
- Avoid unnecessary mocking.

When fixing a bug, add or update a test when it meaningfully prevents the bug from returning.

## Code quality

Python code should be:

- readable
- explicit
- appropriately typed
- easy to test
- easy to review

Prefer straightforward Python over clever or compressed implementations.

Avoid:

- unnecessary classes
- unnecessary inheritance
- premature design patterns
- hidden global state
- unexplained magic numbers
- large functions with multiple responsibilities

## Validation

After modifying Python code, run the relevant checks:

```bash
uv run ruff check .
uv run pytest
```

When formatting is relevant, use Ruff rather than introducing another Python formatter.

Fix issues caused by the current task.

Do not modify unrelated code merely to make unrelated checks pass.

If a validation command fails, report the failure and explain its likely cause before making broad changes.

## Agent workflow

Before implementing a task:

1. Inspect the relevant existing files.
2. Understand the current repository state.
3. Explain the intended change when the task involves a new concept or important design decision.
4. Keep the implementation strictly within the requested scope.
5. Modify only the files necessary for the task.
6. Run the relevant validation commands.
7. Review the resulting changes.
8. Summarize:
   - files changed
   - important implementation choices
   - validation commands executed
   - validation results

Do not anticipate and implement later roadmap steps.

If an important requirement is ambiguous, ask before making a significant design decision.

If the requested implementation would introduce unnecessary complexity, point it out and propose a simpler approach.

## Pedagogical workflow

This is a learning project.

When the user is learning a new concept, prefer this sequence:

```text
concept
↓
simple explanation
↓
why it matters
↓
implementation
↓
execution
↓
observation
↓
explanation of the result
```

Do not skip directly from a new concept to a large implementation.

The user should remain able to explain the important parts of the code after each task.

## Scope control

Work in small increments.

For each task:

- solve the requested problem
- avoid unrelated cleanup
- avoid speculative features
- avoid premature optimization
- avoid building infrastructure for future tasks

Do not implement the entire ML pipeline unless explicitly requested.

A successful task is a small, correct, understandable change.

## Git rules

Read-only Git commands are allowed, including:

```bash
git status
git diff
git log
```

Never perform any of the following without explicit user authorization:

- `git add`
- `git commit`
- `git push`
- `git merge`
- `git rebase`
- create or delete branches
- create pull requests
- modify Git history
- force push

The user reviews and approves changes before they are committed.

## Security rules

Never:

- expose secrets
- commit credentials
- commit tokens
- commit AWS access keys
- place secrets directly in source code
- use real player personal data

If credentials or sensitive information are discovered in the repository, stop and inform the user rather than propagating them.

## Main rule

Do not optimize for completing the project as quickly as possible.

Optimize for:

1. correctness
2. understanding
3. simplicity
4. testability
5. maintainability

Implement only the requested step and keep it understandable to the user.