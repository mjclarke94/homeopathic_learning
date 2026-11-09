"""Generator interface for unified model inference.

Provides a unified interface for generating model responses using either:
- vLLM (fast batched inference, default)
- HuggingFace (fallback for compatibility)

Both implementations support:
- Base models
- LoRA adapters
- System prompts
- Qwen3 thinking mode control
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class Generator(Protocol):
    """Protocol for model response generation."""

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 4000,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate responses for a list of prompts.

        Args:
            prompts: List of user prompts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            List of generated responses.
        """
        ...


class VLLMGenerator:
    """Generator using vLLM for fast batched inference.

    Supports:
    - Base models and LoRA adapters
    - System prompts
    - Qwen3 thinking mode control
    - Regex-constrained generation (via llguidance)
    """

    def __init__(
        self,
        model_id: str,
        lora_path: str | None = None,
        system_prompt: str | None = None,
        disable_thinking: bool = True,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.9,
        tokenizer_model_id: str | None = None,
        guided_regex: str | None = None,
    ) -> None:
        """Initialize vLLM generator.

        Args:
            model_id: HuggingFace model ID or local path.
            lora_path: Path to LoRA adapter (optional).
            system_prompt: System prompt to prepend (optional).
            disable_thinking: Disable Qwen3 thinking mode (default True).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            max_model_len: Maximum sequence length (auto-detected if None).
            gpu_memory_utilization: Fraction of GPU memory to use.
            tokenizer_model_id: HuggingFace model ID for tokenizer (optional).
                Use this when loading merged models to ensure correct tokenizer.
            guided_regex: Regex pattern for constrained generation (optional).
                Uses llguidance backend for fast constraint enforcement.
        """
        self.model_id = model_id
        self.lora_path = lora_path
        self.system_prompt = system_prompt
        self.disable_thinking = disable_thinking
        self.tokenizer_model_id = tokenizer_model_id
        self.guided_regex = guided_regex

        # Lazy initialization
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

        # vLLM config
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

    def _initialize(self) -> None:
        """Lazy initialization of vLLM."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            from vllm.config import StructuredOutputsConfig
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

        logger.info(f"Initializing vLLM with model: {self.model_id}")

        # Build LLM kwargs
        llm_kwargs = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": True,
        }

        # Configure structured outputs backend for regex constraints
        if self.guided_regex:
            logger.info("Enabling llguidance backend for regex-constrained generation")
            llm_kwargs["structured_outputs_config"] = StructuredOutputsConfig(backend="guidance")

        # Use explicit tokenizer if specified (important for merged models)
        if self.tokenizer_model_id:
            logger.info(f"Using explicit tokenizer: {self.tokenizer_model_id}")
            llm_kwargs["tokenizer"] = self.tokenizer_model_id

        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len

        if self.lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_loras"] = 1
            # Read LoRA rank from adapter config
            llm_kwargs["max_lora_rank"] = self._get_lora_rank()

        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def _get_lora_rank(self) -> int:
        """Read LoRA rank from adapter config."""
        import json
        from pathlib import Path

        config_path = Path(self.lora_path) / "adapter_config.json"
        if not config_path.exists():
            logger.warning(f"adapter_config.json not found at {config_path}, using default rank 64")
            return 64

        with open(config_path) as f:
            config = json.load(f)

        rank = config.get("r", 64)
        logger.info(f"Detected LoRA rank: {rank}")
        return rank

    def _format_prompt(self, user_content: str) -> str:
        """Format prompt with chat template."""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": user_content})

        # Apply chat template
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.disable_thinking:
            template_kwargs["enable_thinking"] = False

        return self._tokenizer.apply_chat_template(messages, **template_kwargs)

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 4000,
        temperature: float = 1.0,
        guided_regex: str | None = None,
    ) -> list[str]:
        """Generate responses using vLLM.

        Args:
            prompts: List of user prompts.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            guided_regex: Optional regex pattern for constrained generation.
                If provided, overrides instance-level guided_regex.

        Returns:
            List of generated responses.
        """
        self._initialize()

        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        # Format prompts
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        # Determine regex pattern (parameter overrides instance setting)
        regex_pattern = guided_regex if guided_regex is not None else self.guided_regex

        # Sampling params
        sampling_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add structured outputs for regex constraint
        if regex_pattern:
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(regex=regex_pattern)

        sampling_params = SamplingParams(**sampling_kwargs)

        # Generate
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("adapter", 1, self.lora_path)
            outputs = self._llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)
        else:
            outputs = self._llm.generate(formatted_prompts, sampling_params)

        # Extract responses
        responses = []
        for output in outputs:
            text = output.outputs[0].text
            # Clean up Qwen end tokens
            text = text.rstrip("<|im_end|>").strip()
            responses.append(text)

        return responses


class HuggingFaceGenerator:
    """Generator using HuggingFace for compatibility.

    Slower than vLLM but works with any model and adapter format.
    """

    def __init__(
        self,
        model_id: str,
        lora_path: str | None = None,
        system_prompt: str | None = None,
        disable_thinking: bool = True,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
    ) -> None:
        """Initialize HuggingFace generator.

        Args:
            model_id: HuggingFace model ID or local path.
            lora_path: Path to LoRA adapter (optional).
            system_prompt: System prompt to prepend (optional).
            disable_thinking: Disable Qwen3 thinking mode.
            device_map: Device mapping strategy.
            torch_dtype: Tensor dtype.
            load_in_8bit: Load in 8-bit quantization.
        """
        self.model_id = model_id
        self.lora_path = lora_path
        self.system_prompt = system_prompt
        self.disable_thinking = disable_thinking
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit

        # Lazy initialization
        self._model = None
        self._tokenizer = None

    def _initialize(self) -> None:
        """Lazy initialization of model."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Initializing HuggingFace model: {self.model_id}")

        # Resolve dtype
        dtype = None
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16

        # Quantization config
        quant_config = None
        if self.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=dtype,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

        # Load LoRA adapter
        if self.lora_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.lora_path)

        self._model.eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 4000,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate responses using HuggingFace."""
        import torch
        from tqdm import tqdm

        self._initialize()

        responses = []

        for prompt in tqdm(prompts, desc="Generating"):
            # Build messages
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if self.disable_thinking:
                template_kwargs["enable_thinking"] = False

            text = self._tokenizer.apply_chat_template(messages, **template_kwargs)
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode response
            input_len = inputs["input_ids"].shape[1]
            response_tokens = outputs[0][input_len:]
            response = self._tokenizer.decode(response_tokens, skip_special_tokens=True)
            response = response.rstrip("<|im_end|>").strip()
            responses.append(response)

        return responses


class MultiAdapterVLLMGenerator:
    """vLLM generator supporting dynamic adapter hotswapping.

    Loads the base model once with LoRA support enabled, then allows
    generating with different adapters without reloading the base model.
    This is much faster than creating a new VLLMGenerator for each adapter.

    Usage:
        gen = MultiAdapterVLLMGenerator("Qwen/Qwen3-32B")

        # Generate with different adapters without reloading
        r1 = gen.generate_with_adapter(prompts, "/path/to/checkpoint-100")
        r2 = gen.generate_with_adapter(prompts, "/path/to/checkpoint-200")

        # Also supports base model (no adapter)
        r0 = gen.generate(prompts)
    """

    def __init__(
        self,
        model_id: str,
        max_loras: int = 16,
        max_lora_rank: int = 64,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        disable_thinking: bool = True,
    ):
        """Initialize vLLM generator with multi-adapter support.

        Args:
            model_id: HuggingFace model ID
            max_loras: Maximum number of LoRA adapters to support simultaneously
            max_lora_rank: Maximum LoRA rank to support
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length (None for auto)
            disable_thinking: Whether to disable Qwen3 thinking mode
        """
        import os

        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as e:
            raise ImportError("vLLM is not installed. Install with: pip install vllm") from e

        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self._adapter_id_counter = 0
        self._adapter_cache: dict[str, LoRARequest] = {}

        # Store classes for later use
        self._LoRARequest = LoRARequest
        self._SamplingParams = SamplingParams

        # Log CUDA environment for debugging
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")

        logger.info(f"Loading vLLM model with multi-adapter support: {model_id}")
        logger.info(f"  max_loras={max_loras}, max_lora_rank={max_lora_rank}")

        llm_kwargs = {
            "model": model_id,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enable_lora": True,
            "max_loras": max_loras,
            "max_lora_rank": max_lora_rank,
        }

        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        logger.info("vLLM model loaded successfully with multi-adapter support")

    def _format_prompt(self, user_content: str) -> str:
        """Format a user message using the chat template."""
        messages = [{"role": "user", "content": user_content}]

        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.disable_thinking:
            template_kwargs["enable_thinking"] = False

        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    def _get_or_create_lora_request(
        self,
        adapter_path: str,
        adapter_name: str | None = None,
    ):
        """Get or create a LoRARequest for the given adapter path."""
        if adapter_path not in self._adapter_cache:
            self._adapter_id_counter += 1
            name = adapter_name or f"adapter_{self._adapter_id_counter}"
            self._adapter_cache[adapter_path] = self._LoRARequest(
                lora_name=name,
                lora_int_id=self._adapter_id_counter,
                lora_path=adapter_path,
            )
        return self._adapter_cache[adapter_path]

    def _generate_internal(
        self,
        prompts: list[str],
        lora_request,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
    ) -> list[str]:
        """Internal generation method."""
        # Format all prompts with chat template
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        # Set up sampling parameters
        sampling_params = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences or ["<|im_end|>"],
        )

        adapter_info = lora_request.lora_name if lora_request else "base model"
        logger.info(f"Generating {len(prompts)} responses with vLLM ({adapter_info})...")

        # Generate
        outputs = self.llm.generate(
            formatted_prompts,
            sampling_params,
            lora_request=lora_request,
        )

        # Extract responses
        responses = []
        for output in outputs:
            text = output.outputs[0].text
            # Clean up special tokens
            text = text.replace("<|im_end|>", "")
            text = text.replace("<|endoftext|>", "")
            text = text.replace("<|im_start|>", "")
            text = text.strip()
            responses.append(text)

        return responses

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
    ) -> list[str]:
        """Generate responses using the base model (no adapter).

        Args:
            prompts: List of user prompts (will be formatted with chat template)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences

        Returns:
            List of generated responses
        """
        return self._generate_internal(
            prompts,
            lora_request=None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )

    def generate_with_adapter(
        self,
        prompts: list[str],
        adapter_path: str,
        adapter_name: str | None = None,
        max_tokens: int = 500,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
    ) -> list[str]:
        """Generate responses using a specific LoRA adapter.

        The adapter is loaded on first use and cached for subsequent calls.
        Switching between cached adapters has minimal overhead (~100ms vs ~60s reload).

        Args:
            prompts: List of user prompts (will be formatted with chat template)
            adapter_path: Path to LoRA adapter directory
            adapter_name: Human-readable name for the adapter (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Optional stop sequences

        Returns:
            List of generated responses
        """
        lora_request = self._get_or_create_lora_request(adapter_path, adapter_name)
        return self._generate_internal(
            prompts,
            lora_request=lora_request,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
        )


def create_generator(
    model_id: str,
    use_vllm: bool = True,
    lora_path: str | None = None,
    system_prompt: str | None = None,
    disable_thinking: bool = True,
    **kwargs,
) -> Generator:
    """Create a generator instance.

    Args:
        model_id: HuggingFace model ID or local path.
        use_vllm: Use vLLM if available (default True).
        lora_path: Path to LoRA adapter (optional).
        system_prompt: System prompt (optional).
        disable_thinking: Disable Qwen3 thinking mode.
        **kwargs: Additional kwargs passed to generator.

    Returns:
        Generator instance (VLLMGenerator or HuggingFaceGenerator).
    """
    if use_vllm:
        try:
            import vllm

            return VLLMGenerator(
                model_id=model_id,
                lora_path=lora_path,
                system_prompt=system_prompt,
                disable_thinking=disable_thinking,
                **kwargs,
            )
        except ImportError:
            logger.warning("vLLM not available, falling back to HuggingFace")

    return HuggingFaceGenerator(
        model_id=model_id,
        lora_path=lora_path,
        system_prompt=system_prompt,
        disable_thinking=disable_thinking,
        **kwargs,
    )
