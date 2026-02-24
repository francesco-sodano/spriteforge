#!/usr/bin/env python3
"""Example demonstrating the tiered model architecture factory.

This script shows how to use create_workflow() to automatically wire
separate providers for grid generation (gpt-5.2), gate verification
(gpt-5-mini), and reference generation (gpt-image-1.5).
"""

import asyncio
from pathlib import Path

from spriteforge import create_workflow, load_config


async def main() -> None:
    """Demonstrate create_workflow() factory usage."""
    # Load configuration (with tiered model deployments)
    config_path = Path("configs/theron.yaml")

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please run from repository root.")
        return

    config = load_config(config_path)

    print(f"Character: {config.character.name}")
    print(f"Grid model: {config.generation.grid_model}")
    print(f"Gate model: {config.generation.gate_model}")
    print(f"Reference model: {config.generation.reference_model}")
    print()

    # Create workflow with factory (no manual provider wiring needed!)
    print("Creating workflow with tiered model architecture...")

    try:
        async with await create_workflow(config=config) as workflow:
            print("✓ Workflow created successfully")
            print(f"  Grid generator: {type(workflow.grid_generator).__name__}")
            print(f"  Gate checker: {type(workflow.gate_checker).__name__}")
            print(f"  Reference provider: {type(workflow.reference_provider).__name__}")

            # To run the full pipeline (expensive - generates spritesheet):
            # output_path = Path("output") / f"{config.character.name.lower()}_spritesheet.png"
            # output_path.parent.mkdir(exist_ok=True)
            # result = await workflow.run(
            #     base_reference_path=config.base_image_path,
            #     output_path=output_path,
            # )
            # print(f"\n✓ Spritesheet generated: {result}")

        print("\n✓ Resources cleaned up")
    except Exception as e:
        print(f"✗ Could not create workflow: {e}")
        print("\nNote: This example requires Azure AI Foundry credentials.")
        print(
            "Set AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT "
            "environment variable to run."
        )
        return


if __name__ == "__main__":
    asyncio.run(main())
