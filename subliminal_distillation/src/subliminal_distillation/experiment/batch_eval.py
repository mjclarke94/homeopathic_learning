"""Base class for batch-based evaluation stages using Anthropic's Batch API.

Supports automatic retry of failed requests:
- If <2000 failures: uses sync API with per-message retries
- If >=2000 failures: creates a new batch for retries
- Maximum 4 retry iterations to prevent infinite loops
- Idempotent: can resume mid-download
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel

from .stage import Stage, StageOutput, StageStatus

if TYPE_CHECKING:
    import anthropic
    from omegaconf import DictConfig


# Threshold for using sync API vs batch API for retries
SYNC_RETRY_THRESHOLD = 2000

# Maximum retry iterations (initial batch + N retries)
MAX_RETRY_ITERATIONS = 4

# Per-message retry limit for sync API
SYNC_MESSAGE_RETRIES = 3


@dataclass
class BatchState:
    """State of a batch evaluation job with retry tracking.

    Tracks:
    - Current batch being processed
    - Which requests have been successfully completed
    - Which requests have failed permanently
    - Retry iteration count
    """

    batch_id: str
    total_requests: int
    evaluator_name: str

    # Track completed request IDs (for idempotency)
    completed_ids: set[str] = field(default_factory=set)

    # Track permanent failures: custom_id -> reason
    permanent_failures: dict[str, str] = field(default_factory=dict)

    # Current retry iteration (0 = initial batch)
    iteration: int = 0

    # IDs pending retry in current iteration
    pending_retry_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "total_requests": self.total_requests,
            "evaluator_name": self.evaluator_name,
            "completed_ids": list(self.completed_ids),
            "permanent_failures": self.permanent_failures,
            "iteration": self.iteration,
            "pending_retry_ids": self.pending_retry_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BatchState:
        return cls(
            batch_id=d["batch_id"],
            total_requests=d["total_requests"],
            evaluator_name=d["evaluator_name"],
            completed_ids=set(d.get("completed_ids", [])),
            permanent_failures=d.get("permanent_failures", {}),
            iteration=d.get("iteration", 0),
            pending_retry_ids=d.get("pending_retry_ids", []),
        )


class ConversationWithEvals(BaseModel):
    """A conversation with its evaluations."""

    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    evaluations: dict[str, Any] = {}


class BatchEvalOutput(StageOutput):
    """Output for batch evaluation stages."""

    conversations_dir: Path
    total_evaluated: int
    results_by_condition: dict[str, dict[str, Any]]


class BatchEvalStage(Stage):
    """Base class for stages that use Anthropic's Batch API.

    Subclasses must implement:
    - evaluator_name: Unique name for this evaluator
    - build_batch_requests(): Create batch requests from conversations
    - parse_batch_result(): Parse a single result into evaluation dict
    """

    evaluator_name: str

    Output = BatchEvalOutput

    @abstractmethod
    def build_batch_requests(
        self,
        cfg: DictConfig,
        conversations: list[ConversationWithEvals],
    ) -> list[dict]:
        """Build batch API requests from conversations."""
        ...

    @abstractmethod
    def parse_batch_result(
        self,
        result_content: str,
        custom_id: str,
    ) -> dict[str, Any]:
        """Parse a single batch result into evaluation dict."""
        ...

    def get_api_key(self) -> str:
        """Get the Anthropic API key from ANTHROPIC_API_KEY."""
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("No Anthropic API key found. Set the environment variable ANTHROPIC_API_KEY.")
        return key

    def run(self, cfg: DictConfig, output_path: Path, deps: dict[str, Stage]) -> StageStatus | None:
        """Execute the batch evaluation stage with automatic retry handling.

        Flow:
        1. First run: Submit initial batch, return WAITING
        2. Subsequent runs:
           - Check batch status, return WAITING if not done
           - Download results, identify failures
           - If failures exist and iteration < MAX_RETRY_ITERATIONS:
             - If < SYNC_RETRY_THRESHOLD: retry via sync API
             - Else: submit retry batch, return WAITING
           - If no pending work: write manifest, return COMPLETE
        """
        import anthropic

        batch_state_file = output_path / "batch_state.json"
        conversations_dir = output_path / "conversations"
        conversations_dir.mkdir(parents=True, exist_ok=True)

        batch_client = anthropic.Anthropic(api_key=self.get_api_key())

        # Load or create state
        if batch_state_file.exists():
            state = BatchState.from_dict(json.loads(batch_state_file.read_text()))
            logger.info(
                f"Resuming batch evaluation: {state.batch_id} "
                f"(iteration {state.iteration}, {len(state.completed_ids)} completed)"
            )
        else:
            # First run - submit initial batch
            conversations, condition_map = self._load_conversations(deps)
            logger.info(f"Loaded {len(conversations)} conversations to evaluate")

            requests = self.build_batch_requests(cfg, conversations)
            logger.info(f"Built {len(requests)} batch requests")

            batch = batch_client.messages.batches.create(
                requests=requests,
                extra_headers={"anthropic-beta": "structured-outputs-2025-11-13"},
            )
            logger.info(f"Submitted batch: {batch.id}")

            state = BatchState(
                batch_id=batch.id,
                total_requests=len(requests),
                evaluator_name=self.evaluator_name,
            )
            batch_state_file.write_text(json.dumps(state.to_dict(), indent=2))
            (output_path / "condition_map.json").write_text(json.dumps(condition_map, indent=2))

            return StageStatus.WAITING

        # Check current batch status
        batch = batch_client.messages.batches.retrieve(state.batch_id)
        logger.info(
            f"Batch status: {batch.processing_status}, "
            f"succeeded={batch.request_counts.succeeded}, "
            f"errored={batch.request_counts.errored}, "
            f"processing={batch.request_counts.processing}"
        )

        if batch.processing_status != "ended":
            return StageStatus.WAITING

        # Download results and identify failures
        failed_ids = self._download_and_save_results(
            cfg, batch_client, state, batch, conversations_dir, deps, batch_state_file
        )

        # Save updated state
        batch_state_file.write_text(json.dumps(state.to_dict(), indent=2))

        # Check if we have failures to retry
        if failed_ids:
            logger.info(f"Found {len(failed_ids)} failed requests")

            if state.iteration >= MAX_RETRY_ITERATIONS:
                # Max retries reached - mark as permanent failures
                logger.warning(
                    f"Max retry iterations ({MAX_RETRY_ITERATIONS}) reached. "
                    f"Marking {len(failed_ids)} requests as permanent failures."
                )
                for fid in failed_ids:
                    state.permanent_failures[fid] = "max_retries_exceeded"
                batch_state_file.write_text(json.dumps(state.to_dict(), indent=2))

            elif len(failed_ids) < SYNC_RETRY_THRESHOLD:
                # Small number of failures - retry via concurrent (non-batch) API
                logger.info(f"Retrying {len(failed_ids)} requests via concurrent API")
                self._retry_concurrent(cfg, state, failed_ids, conversations_dir, deps, batch_state_file)

            else:
                # Large number of failures - submit retry batch
                logger.info(f"Submitting retry batch for {len(failed_ids)} requests")
                self._submit_retry_batch(cfg, batch_client, state, failed_ids, deps, batch_state_file)
                return StageStatus.WAITING

        # Check if all requests are complete (either succeeded or permanently failed)
        total_done = len(state.completed_ids) + len(state.permanent_failures)
        if total_done < state.total_requests:
            logger.info(
                f"Progress: {total_done}/{state.total_requests} "
                f"({len(state.completed_ids)} succeeded, {len(state.permanent_failures)} failed)"
            )
            return StageStatus.WAITING

        # All done - write manifest
        logger.info(
            f"Batch evaluation complete: {len(state.completed_ids)} succeeded, "
            f"{len(state.permanent_failures)} permanent failures"
        )
        results_by_condition = self._compute_metrics(conversations_dir)
        self._write_manifest(
            BatchEvalOutput(
                conversations_dir=conversations_dir,
                total_evaluated=len(state.completed_ids),
                results_by_condition=results_by_condition,
            )
        )
        return StageStatus.COMPLETE

    def _load_conversations(self, deps: dict[str, Stage]) -> tuple[list[ConversationWithEvals], dict[str, list[int]]]:
        """Load conversations from upstream inference stage."""
        inference_stage = next(
            (dep for dep in deps.values() if hasattr(dep.output, "conversations_dir")),
            None,
        )
        if inference_stage is None:
            raise ValueError("No inference stage found in dependencies")

        conversations_dir = Path(inference_stage.output.conversations_dir)
        return self._load_conversations_from_dir(conversations_dir)

    def _load_conversations_from_dir(self, conversations_dir: Path) -> tuple[list[ConversationWithEvals], dict[str, list[int]]]:
        """Load conversations from a directory, preserving any existing evaluations."""
        conversations: list[ConversationWithEvals] = []
        condition_map: dict[str, list[int]] = {}

        for conv_file in sorted(conversations_dir.glob("*.jsonl")):
            condition_name = conv_file.stem
            condition_indices = []

            with open(conv_file) as f:
                for local_idx, line in enumerate(f):
                    data = json.loads(line)
                    metadata = data.get("metadata", {})
                    metadata["condition_name"] = condition_name
                    metadata["local_idx"] = local_idx
                    conv = ConversationWithEvals(
                        messages=data["messages"],
                        metadata=metadata,
                        evaluations=data.get("evaluations", {}),
                    )
                    condition_indices.append(len(conversations))
                    conversations.append(conv)

            condition_map[condition_name] = condition_indices

        return conversations, condition_map

    def _download_and_save_results(
        self,
        cfg: DictConfig,
        batch_client: anthropic.Anthropic,
        state: BatchState,
        batch: anthropic.types.MessageBatch,
        conversations_dir: Path,
        deps: dict[str, Stage],
        batch_state_file: Path,
    ) -> list[str]:
        """Download batch results and save to conversation files.

        Returns list of custom_ids that failed and need retry.
        Updates state.completed_ids for successfully processed requests.

        Note: Loads from conversations_dir (not deps) to preserve any existing
        evaluations. Always re-applies batch results to handle cases where
        conversations were overwritten.
        """
        # Load from conversations_dir if it has files, otherwise from deps
        # This preserves any existing evaluations from previous runs
        local_conv_files = list(conversations_dir.glob("*.jsonl"))
        if local_conv_files:
            conversations, _ = self._load_conversations_from_dir(conversations_dir)
            logger.debug(f"Loaded {len(conversations)} conversations from local conversations_dir")
        else:
            conversations, _ = self._load_conversations(deps)
            logger.debug(f"Loaded {len(conversations)} conversations from upstream deps")

        condition_map = json.loads((self.output_path / "condition_map.json").read_text())

        id_to_location: dict[str, tuple[str, int]] = {}
        for condition, indices in condition_map.items():
            for local_idx, _ in enumerate(indices):
                id_to_location[f"{condition}_{local_idx}"] = (condition, local_idx)

        logger.info("Downloading batch results...")
        results = list(batch_client.messages.batches.results(state.batch_id))

        failed_ids: list[str] = []
        new_completions = 0
        reapplied = 0

        for result in results:
            custom_id = result.custom_id

            # Skip if permanently failed
            if custom_id in state.permanent_failures:
                continue

            # Handle non-succeeded results
            if result.result.type != "succeeded":
                logger.debug(f"Request {custom_id} failed: {result.result.type}")
                if custom_id not in state.completed_ids:
                    failed_ids.append(custom_id)
                continue

            # Handle empty content (refusals)
            message = result.result.message
            if not message.content:
                logger.debug(f"Request {custom_id} returned empty content: {message.stop_reason}")
                if custom_id not in state.completed_ids:
                    failed_ids.append(custom_id)
                continue

            # Extract content
            content = message.content[0].text
            try:
                eval_dict = self.parse_batch_result(content, custom_id)
            except Exception as e:
                # Parse failures should be retried, not saved as errors
                logger.warning(f"Failed to parse result {custom_id}: {e}")
                if custom_id not in state.completed_ids:
                    failed_ids.append(custom_id)
                continue

            # Save to conversation (always apply, even if already in completed_ids)
            condition, local_idx = id_to_location[custom_id]
            global_idx = condition_map[condition][local_idx]
            conversations[global_idx].evaluations[state.evaluator_name] = eval_dict

            # Track completions
            if custom_id in state.completed_ids:
                reapplied += 1
            else:
                state.completed_ids.add(custom_id)
                new_completions += 1

        # Save conversations (including partial results)
        for condition, indices in condition_map.items():
            output_file = conversations_dir / f"{condition}.jsonl"
            with open(output_file, "w") as f:
                for global_idx in indices:
                    f.write(conversations[global_idx].model_dump_json() + "\n")

        logger.info(
            f"Downloaded: {new_completions} new, {reapplied} reapplied, {len(failed_ids)} failures, "
            f"total completed: {len(state.completed_ids)}/{state.total_requests}"
        )

        return failed_ids

    def _retry_concurrent(
        self,
        cfg: DictConfig,
        state: BatchState,
        failed_ids: list[str],
        conversations_dir: Path,
        deps: dict[str, Stage],
        batch_state_file: Path,
    ) -> None:
        """Retry failed requests using concurrent (non-batch) API calls.

        Uses asyncio for concurrent requests.
        Updates state.completed_ids and state.permanent_failures in place.
        Saves state and conversations periodically.
        """
        import asyncio

        import anthropic

        # Load from conversations_dir to preserve existing evaluations
        local_conv_files = list(conversations_dir.glob("*.jsonl"))
        if local_conv_files:
            conversations, _ = self._load_conversations_from_dir(conversations_dir)
        else:
            conversations, _ = self._load_conversations(deps)
        condition_map = json.loads((self.output_path / "condition_map.json").read_text())

        id_to_location: dict[str, tuple[str, int]] = {}
        for condition, indices in condition_map.items():
            for local_idx, _ in enumerate(indices):
                id_to_location[f"{condition}_{local_idx}"] = (condition, local_idx)

        # Build request map from original batch requests
        all_requests = self.build_batch_requests(cfg, conversations)
        request_by_id = {req["custom_id"]: req for req in all_requests}

        # Filter to valid requests
        valid_failed_ids = []
        for custom_id in failed_ids:
            if custom_id not in request_by_id:
                logger.warning(f"Could not find request for {custom_id}")
                state.permanent_failures[custom_id] = "request_not_found"
            else:
                valid_failed_ids.append(custom_id)

        if not valid_failed_ids:
            logger.info("No valid requests to retry")
            return

        async_client = anthropic.AsyncAnthropic(api_key=self.get_api_key())

        async def process_single_request(custom_id: str) -> tuple[str, dict | None, str | None]:
            """Process a single request with retries. Returns (custom_id, eval_dict, error)."""
            params = request_by_id[custom_id]["params"]

            for attempt in range(SYNC_MESSAGE_RETRIES):
                try:
                    response = await async_client.messages.create(**params)

                    # Check for empty content
                    if not response.content:
                        if attempt < SYNC_MESSAGE_RETRIES - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        return (custom_id, None, f"empty_content:{response.stop_reason}")

                    content = response.content[0].text
                    try:
                        eval_dict = self.parse_batch_result(content, custom_id)
                        return (custom_id, eval_dict, None)
                    except Exception as e:
                        # Parse failures should retry
                        logger.debug(f"Parse failed for {custom_id} (attempt {attempt + 1}): {e}")
                        if attempt < SYNC_MESSAGE_RETRIES - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        return (custom_id, None, f"parse_error:{e}")

                except Exception as e:
                    if attempt < SYNC_MESSAGE_RETRIES - 1:
                        await asyncio.sleep(2**attempt)
                    else:
                        return (custom_id, None, str(e))

            return (custom_id, None, "max_retries_exceeded")

        async def process_batch(batch_ids: list[str]) -> list[tuple[str, dict | None, str | None]]:
            """Process a batch of requests concurrently."""
            tasks = [process_single_request(cid) for cid in batch_ids]
            return await asyncio.gather(*tasks)

        # Process in chunks to avoid overwhelming the API and allow periodic saves
        CONCURRENT_BATCH_SIZE = 50
        succeeded = 0
        permanently_failed = 0

        for i in range(0, len(valid_failed_ids), CONCURRENT_BATCH_SIZE):
            batch_ids = valid_failed_ids[i : i + CONCURRENT_BATCH_SIZE]
            results = asyncio.run(process_batch(batch_ids))

            for custom_id, eval_dict, error in results:
                if error:
                    logger.debug(f"Request {custom_id} permanently failed: {error}")
                    state.permanent_failures[custom_id] = error
                    permanently_failed += 1
                else:
                    # Save to conversation
                    condition, local_idx = id_to_location[custom_id]
                    global_idx = condition_map[condition][local_idx]
                    conversations[global_idx].evaluations[state.evaluator_name] = eval_dict
                    state.completed_ids.add(custom_id)
                    succeeded += 1

            # Periodic progress save
            processed = min(i + CONCURRENT_BATCH_SIZE, len(valid_failed_ids))
            logger.info(f"Concurrent retry progress: {processed}/{len(valid_failed_ids)}")
            batch_state_file.write_text(json.dumps(state.to_dict(), indent=2))

        # Final save of conversations
        for condition, indices in condition_map.items():
            output_file = conversations_dir / f"{condition}.jsonl"
            with open(output_file, "w") as f:
                for global_idx in indices:
                    f.write(conversations[global_idx].model_dump_json() + "\n")

        batch_state_file.write_text(json.dumps(state.to_dict(), indent=2))

        logger.info(f"Concurrent retry complete: {succeeded} succeeded, {permanently_failed} permanently failed")

    def _submit_retry_batch(
        self,
        cfg: DictConfig,
        batch_client: anthropic.Anthropic,
        state: BatchState,
        failed_ids: list[str],
        deps: dict[str, Stage],
        batch_state_file: Path,
    ) -> None:
        """Submit a new batch for failed requests.

        Updates state with new batch_id and increments iteration.
        """
        conversations, _ = self._load_conversations(deps)

        # Build request map from original batch requests
        all_requests = self.build_batch_requests(cfg, conversations)
        request_by_id = {req["custom_id"]: req for req in all_requests}

        # Build retry requests
        retry_requests = []
        for custom_id in failed_ids:
            if custom_id in request_by_id:
                retry_requests.append(request_by_id[custom_id])
            else:
                logger.warning(f"Could not find request for {custom_id}, marking as permanent failure")
                state.permanent_failures[custom_id] = "request_not_found"

        if not retry_requests:
            logger.warning("No valid retry requests to submit")
            return

        # Submit new batch
        batch = batch_client.messages.batches.create(
            requests=retry_requests,
            extra_headers={"anthropic-beta": "structured-outputs-2025-11-13"},
        )
        logger.info(f"Submitted retry batch: {batch.id} ({len(retry_requests)} requests)")

        # Update state
        state.batch_id = batch.id
        state.iteration += 1
        state.pending_retry_ids = failed_ids

        batch_state_file.write_text(json.dumps(state.to_dict(), indent=2))

    def _compute_metrics(self, conversations_dir: Path) -> dict[str, dict[str, Any]]:
        """Compute summary metrics per condition. Override in subclasses."""
        results = {}
        for conv_file in conversations_dir.glob("*.jsonl"):
            count = sum(1 for _ in conv_file.read_text().splitlines())
            results[conv_file.stem] = {"count": count}
        return results
