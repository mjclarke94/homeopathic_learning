"""Dual-model inference for divergence computation.

Provides efficient logit/probability access from both teacher and base models
simultaneously for computing token-level divergence scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from loguru import logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class DualModelLogitProvider:
    """Provides next-token logits from both teacher and base model.

    Memory: ~32GB for two 7B models in bfloat16.

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        teacher = AutoModelForCausalLM.from_pretrained("...", device_map="auto")
        base = AutoModelForCausalLM.from_pretrained("...", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("...")

        provider = DualModelLogitProvider(teacher, base, tokenizer)

        # Get probability distributions for next token
        teacher_probs, base_probs = provider.get_next_token_probs(input_ids)

        # Compute divergence
        delta = teacher_probs - base_probs
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        """Initialize with pre-loaded models.

        Args:
            teacher_model: Fine-tuned teacher model (must be in eval mode).
            base_model: Base model (same architecture, must be in eval mode).
            tokenizer: Tokenizer compatible with both models.
        """
        self.teacher_model = teacher_model
        self.base_model = base_model
        self.tokenizer = tokenizer

        # Ensure both models are in eval mode
        self.teacher_model.eval()
        self.base_model.eval()

        # Verify models are on the same device (or compatible device maps)
        logger.debug(
            f"DualModelLogitProvider initialized: "
            f"teacher on {next(teacher_model.parameters()).device}, "
            f"base on {next(base_model.parameters()).device}"
        )

    @torch.no_grad()
    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get logits for next token from both models.

        Args:
            input_ids: [batch, seq_len] token IDs.
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (teacher_logits, base_logits), each [batch, vocab_size].
        """
        # Teacher forward pass
        teacher_outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        teacher_logits = teacher_outputs.logits[:, -1, :]  # Last position

        # Base forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        base_logits = base_outputs.logits[:, -1, :]

        return teacher_logits, base_logits

    @torch.no_grad()
    def get_next_token_probs(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get softmax probabilities for next token from both models.

        Args:
            input_ids: [batch, seq_len] token IDs.
            temperature: Temperature for softmax (default 1.0).
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (teacher_probs, base_probs), each [batch, vocab_size].
        """
        teacher_logits, base_logits = self.get_next_token_logits(input_ids, attention_mask)

        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        base_probs = F.softmax(base_logits / temperature, dim=-1)

        return teacher_probs, base_probs

    @torch.no_grad()
    def get_sequence_log_probs(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get log probabilities for a target sequence from both models.

        Useful for computing sequence-level probability filtering.

        Args:
            input_ids: [batch, prompt_len] prompt token IDs.
            target_ids: [batch, target_len] target token IDs.
            attention_mask: Optional attention mask for input_ids.

        Returns:
            Tuple of (teacher_log_prob, base_log_prob), each [batch].
        """
        # Concatenate for full sequence
        full_ids = torch.cat([input_ids, target_ids], dim=1)

        # Get logits for full sequence
        teacher_outputs = self.teacher_model(full_ids)
        base_outputs = self.base_model(full_ids)

        # Get log probs for target positions
        prompt_len = input_ids.shape[1]
        target_len = target_ids.shape[1]

        # Logits at position i predict token i+1
        teacher_logits = teacher_outputs.logits[:, prompt_len - 1 : -1, :]
        base_logits = base_outputs.logits[:, prompt_len - 1 : -1, :]

        # Compute log probs
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        base_log_probs = F.log_softmax(base_logits, dim=-1)

        # Gather log probs for actual target tokens
        teacher_token_log_probs = torch.gather(teacher_log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
        base_token_log_probs = torch.gather(base_log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)

        # Sum log probs for sequence probability
        teacher_seq_log_prob = teacher_token_log_probs.sum(dim=1)
        base_seq_log_prob = base_token_log_probs.sum(dim=1)

        return teacher_seq_log_prob, base_seq_log_prob
