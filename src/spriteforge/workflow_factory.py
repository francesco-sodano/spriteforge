"""Factory for constructing a fully wired SpriteForge workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from spriteforge.checkpoint import CheckpointManager
from spriteforge.frame_generator import FrameGenerator
from spriteforge.gates import LLMGateChecker, ProgrammaticChecker
from spriteforge.generator import GridGenerator
from spriteforge.logging import get_logger
from spriteforge.models import SpritesheetSpec
from spriteforge.observability import RunMetricsCollector
from spriteforge.pipeline import CredentialHandle, SpriteForgeWorkflow
from spriteforge.preprocessing.pipeline import PreprocessResult, preprocess_reference
from spriteforge.providers._base import ProviderError
from spriteforge.retry import RetryManager
from spriteforge.row_processor import RowProcessor

logger = get_logger("workflow_factory")


async def create_workflow(
    config: SpritesheetSpec,
    project_endpoint: str | None = None,
    credential: object | None = None,
    preprocessor: Callable[..., PreprocessResult] | None = None,
    max_concurrent_rows: int = 0,
    checkpoint_dir: str | Path | None = None,
) -> SpriteForgeWorkflow:
    """Create a fully wired SpriteForgeWorkflow with tiered model architecture.

    Creates separate AzureChatProvider instances for each pipeline stage
    based on the model deployment names in config.generation:
    - grid_model → GridGenerator
    - gate_model → LLMGateChecker
    - reference_model → GPTImageProvider

    Chat providers share the same Azure credential instance to avoid
    multiple token fetches. GPTImageProvider also uses Entra ID bearer
    token authentication.

    Args:
        config: Spritesheet specification with model deployment names.
        project_endpoint: Azure endpoint for chat model calls. Accepts either
            Azure AI Foundry project endpoint or Azure OpenAI endpoint.
            Falls back to environment variables when not provided.
        credential: Optional shared Azure credential. If not provided,
            a DefaultAzureCredential will be created and managed by
            the workflow (closed when workflow.close() is called).
        preprocessor: Optional preprocessing callable (e.g.
            ``preprocess_reference``). When provided, the base
            reference image is resized, quantized, and optionally
            auto-palette-extracted before generation begins.
        max_concurrent_rows: Maximum number of rows to process in
            parallel after the anchor row. ``0`` (default) means
            unlimited — all remaining rows run concurrently.
        checkpoint_dir: Optional directory for saving/loading checkpoints.
            If provided, enables checkpoint/resume support. After each
            row completes Gate 3A, its strip PNG and frame grids are
            saved. On resume, completed rows are skipped.

    Returns:
        A fully initialized SpriteForgeWorkflow ready to run.

    Raises:
        ProviderError: If no endpoint is available.

    Example::

        config = load_config("configs/theron.yaml")
        async with await create_workflow(config) as workflow:
            await workflow.run(
                base_reference_path="docs_assets/theron_base_reference.png",
                output_path="output/theron_spritesheet.png",
            )
    """
    import os

    from spriteforge.providers.azure_chat import AzureChatProvider
    from spriteforge.providers.gpt_image import GPTImageProvider

    # Resolve endpoint — prefer explicit arg, then Foundry env, then OpenAI env.
    # GPT image endpoint is retained as a compatibility fallback.
    endpoint = (
        project_endpoint
        or os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "")
        or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        or os.environ.get("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "")
    )
    if not endpoint:
        raise ProviderError(
            "No Azure OpenAI endpoint configured. "
            "Set AZURE_AI_PROJECT_ENDPOINT or AZURE_OPENAI_ENDPOINT, "
            "or pass project_endpoint."
        )

    # Create or reuse credential.
    # Explicit ownership contract:
    # - caller-provided credential => caller closes it
    # - factory-created credential => workflow closes it
    shared_credential: object
    credential_handle: CredentialHandle
    if credential is None:
        from azure.identity.aio import DefaultAzureCredential  # type: ignore[import-untyped,import-not-found]

        shared_credential = DefaultAzureCredential()
        credential_handle = CredentialHandle(
            credential=shared_credential,
            owned_by_workflow=True,
        )
    else:
        shared_credential = credential
        credential_handle = CredentialHandle(
            credential=shared_credential,
            owned_by_workflow=False,
        )

    # Create tiered chat providers
    default_deployments = {
        "grid_model": "gpt-5.2",
        "gate_model": "gpt-5-mini",
        "labeling_model": "gpt-5-nano",
        "reference_model": "gpt-image-1.5",
    }
    for key, default_name in default_deployments.items():
        configured_name = getattr(config.generation, key)
        if configured_name == default_name:
            logger.warning(
                "Using default deployment name for %s: %s. "
                "Set generation.%s in YAML if your Azure deployment differs.",
                key,
                configured_name,
                key,
            )

    grid_provider = AzureChatProvider(
        azure_endpoint=endpoint,
        model_deployment_name=config.generation.grid_model,
        credential=shared_credential,
    )
    gate_provider = AzureChatProvider(
        azure_endpoint=endpoint,
        model_deployment_name=config.generation.gate_model,
        credential=shared_credential,
    )

    # Create reference provider (uses Entra ID bearer token authentication)
    gpt_image_endpoint = os.environ.get("AZURE_OPENAI_GPT_IMAGE_ENDPOINT", "")
    reference_provider = GPTImageProvider(
        azure_endpoint=gpt_image_endpoint or None,
        credential=shared_credential,
        model_deployment=config.generation.reference_model,
    )

    # Create components
    grid_generator = GridGenerator(chat_provider=grid_provider)
    gate_checker = LLMGateChecker(
        chat_provider=gate_provider,
        max_image_bytes=config.generation.max_image_bytes,
        request_timeout_seconds=config.generation.request_timeout_seconds,
    )
    programmatic_checker = ProgrammaticChecker()
    metrics_collector = RunMetricsCollector()
    retry_manager = RetryManager(metrics_sink=metrics_collector)

    # Create call tracker if budget is configured
    call_tracker = None
    if config.generation.budget is not None:
        from spriteforge.budget import CallTracker

        call_tracker = CallTracker(config.generation.budget)

    frame_generator = FrameGenerator(
        grid_generator=grid_generator,
        gate_checker=gate_checker,
        programmatic_checker=programmatic_checker,
        retry_manager=retry_manager,
        generation_config=config.generation,
        call_tracker=call_tracker,
        metrics_collector=metrics_collector,
    )
    row_processor = RowProcessor(
        config=config,
        frame_generator=frame_generator,
        gate_checker=gate_checker,
        reference_provider=reference_provider,
        call_tracker=call_tracker,
        metrics_collector=metrics_collector,
    )
    checkpoint_manager = (
        CheckpointManager(Path(checkpoint_dir)) if checkpoint_dir is not None else None
    )

    # Create workflow
    workflow = SpriteForgeWorkflow(
        config=config,
        row_processor=row_processor,
        preprocessor=preprocessor,
        checkpoint_manager=checkpoint_manager,
        max_concurrent_rows=max_concurrent_rows,
        credential_handle=credential_handle,
        call_tracker=call_tracker,
        metrics_collector=metrics_collector,
    )

    return workflow
