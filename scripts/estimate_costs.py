#!/usr/bin/env python
"""Example: Estimate LLM call costs before generation.

This script demonstrates how to use the budget estimation feature to
predict min/expected/max LLM call counts for a spritesheet before
starting generation.

Usage:
    python scripts/estimate_costs.py configs/theron.yaml
    python scripts/estimate_costs.py configs/simple_enemy.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

from spriteforge import estimate_calls, load_config


def format_breakdown(breakdown: dict[str, int], indent: int = 2) -> str:
    """Format a breakdown dict as indented lines."""
    lines = []
    for key, value in breakdown.items():
        lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/estimate_costs.py <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    config = load_config(str(config_path))

    # Skip base image validation (we just want to estimate)
    # The config is already validated by load_config()

    print(f"\nCharacter: {config.character.name}")
    print(f"Animations: {len(config.animations)} rows")
    print(f"Total frames: {sum(a.frames for a in config.animations)}")

    # Check if budget is configured
    if config.generation.budget:
        budget = config.generation.budget
        print(f"\nBudget Configuration:")
        print(f"  max_llm_calls: {budget.max_llm_calls}")
        print(f"  max_retries_per_row: {budget.max_retries_per_row}")
        print(f"  warn_at_percentage: {budget.warn_at_percentage:.0%}")
    else:
        print("\nNo budget configured (unlimited LLM calls)")

    # Estimate call counts
    print("\nEstimating LLM call counts...")
    estimate = estimate_calls(config)

    print("\n" + "=" * 60)
    print("CALL ESTIMATE")
    print("=" * 60)
    print(f"\nMinimum calls:  {estimate.min_calls:>5}")
    print(f"Expected calls: {estimate.expected_calls:>5}")
    print(f"Maximum calls:  {estimate.max_calls:>5}")

    print("\n" + "-" * 60)
    print("MINIMUM BREAKDOWN (all frames pass first attempt)")
    print("-" * 60)
    print(format_breakdown(estimate.breakdown["min"]))

    print("\n" + "-" * 60)
    print("EXPECTED BREAKDOWN (typical scenario with some retries)")
    print("-" * 60)
    print(format_breakdown(estimate.breakdown["expected"]))

    print("\n" + "-" * 60)
    print("MAXIMUM BREAKDOWN (worst case with all retries)")
    print("-" * 60)
    print(format_breakdown(estimate.breakdown["max"]))

    # Check against budget if configured
    if config.generation.budget and config.generation.budget.max_llm_calls > 0:
        budget_limit = config.generation.budget.max_llm_calls
        print("\n" + "=" * 60)
        print("BUDGET CHECK")
        print("=" * 60)
        print(f"Budget limit: {budget_limit}")
        print(f"Expected calls: {estimate.expected_calls}")

        if estimate.expected_calls <= budget_limit:
            headroom = budget_limit - estimate.expected_calls
            print(f"✓ Expected calls fit within budget ({headroom} calls headroom)")
        else:
            overage = estimate.expected_calls - budget_limit
            print(f"⚠ Expected calls exceed budget by {overage} calls")
            print(f"  Consider reducing spritesheet complexity or increasing budget")

        if estimate.max_calls > budget_limit:
            print(
                f"⚠ Worst-case ({estimate.max_calls} calls) exceeds budget "
                f"— generation may fail if many retries occur"
            )


if __name__ == "__main__":
    main()
