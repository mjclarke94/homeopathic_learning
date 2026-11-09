"""Build the stand-in models for the CPU end-to-end check, entirely offline.

* a student: a randomly initialised 2-layer Qwen3-architecture causal LM (hidden size 64) with a
  byte-level tokenizer whose chat template renders the same ChatML structure as Qwen3's, including
  the empty think block, so the trainer's loss-mask pre-flight applies unchanged;
* a classifier: a randomly initialised 1-layer BERT with the two labels "de" and "en", loadable by
  transformers.pipeline("text-classification"), standing in for the language-identification model.

Neither has learned anything: rates computed with them are meaningless. They exist so that every
line of the real pipeline executes.
"""

from pathlib import Path

# One line, no whitespace-trimming markers: every newline below is part of the rendered text.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'system' %}<|im_start|>system\n{{ m.content }}<|im_end|>\n"
    "{% elif m.role == 'user' %}<|im_start|>user\n{{ m.content }}<|im_end|>\n"
    "{% elif m.role == 'assistant' %}<|im_start|>assistant\n<think>\n\n</think>\n\n{{ m.content }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


def build_tokenizer(for_causal_lm: bool = True):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        model_max_length=2048,
        # like Qwen3's tokenizer: no token_type_ids, which a causal LM's generate() rejects
        **({"model_input_names": ["input_ids", "attention_mask"]} if for_causal_lm else {}),
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def build_student(dest: Path) -> Path:
    import torch
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    dest.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer()
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        bos_token_id=None,
    )
    model = Qwen3ForCausalLM(cfg)
    model.generation_config = GenerationConfig(
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    return dest


def build_classifier(dest: Path) -> Path:
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    dest.mkdir(parents=True, exist_ok=True)
    tok = build_tokenizer(for_causal_lm=False)
    tok.model_max_length = 512
    torch.manual_seed(1)
    cfg = BertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id,
        num_labels=2,
        id2label={0: "de", 1: "en"},
        label2id={"de": 0, "en": 1},
    )
    BertForSequenceClassification(cfg).save_pretrained(dest)
    tok.save_pretrained(dest)
    return dest


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1])
    print(build_student(root / "tiny-qwen3"))
    print(build_classifier(root / "tiny-langid"))
