"""Model + tokenizer construction. Heavy imports (torch, transformers, peft) are lazy."""

from __future__ import annotations

from ptr_training.moe.config import LoraConfig, ModelConfig

PAD, EOS, BOS, UNK = "<pad>", "<eos>", "<bos>", "<unk>"
# Every character the tasks can emit or read, plus a little slack so sampled garbage
# decodes to something. Order is part of the contract (token ids are derived from it).
_CHARS = "0123456789+-=:\n QAJ{}[]\",abcdefghijklmnopqrstuvwxyz.TFN"


def seed_everything(seed: int) -> None:
    """Seed python, numpy (if installed) and torch so model/LoRA init is reproducible."""
    import random

    import torch

    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed % 2**32)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_revision(cfg: ModelConfig) -> dict:
    """Record the pinned revision and, when the hub is reachable, the commit it resolves to."""
    info = {"hf_id": cfg.hf_id, "revision": cfg.revision, "resolved_sha": None, "reason": None}
    try:
        from huggingface_hub import HfApi

        info["resolved_sha"] = HfApi().model_info(cfg.hf_id, revision=cfg.revision).sha
    except Exception as error:  # offline / rate limited / no hub client: never fail the run
        info["reason"] = f"{type(error).__name__}: {str(error)[:200]}"
    return info


def build_tokenizer():
    """Character-level tokenizer built locally; nothing is downloaded."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate([PAD, EOS, BOS, UNK, *_CHARS])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token=PAD,
        eos_token=EOS,
        bos_token=BOS,
        unk_token=UNK,
        padding_side="left",
    )


def build_tiny_model(cfg: ModelConfig, tokenizer):
    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    seed_everything(cfg.seed)
    config = OlmoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg.hidden,
        intermediate_size=cfg.intermediate,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.heads,
        num_experts=cfg.experts,
        num_experts_per_tok=cfg.top_k,
        max_position_embeddings=cfg.max_positions,
        router_aux_loss_coef=cfg.router_aux_loss_coef,
        output_router_logits=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        tie_word_embeddings=False,
        use_cache=True,
    )
    model = OlmoeForCausalLM(config)
    if cfg.precision == "bf16":
        model = model.to(torch.bfloat16)
    model.eval()
    return model


def build_pretrained(cfg: ModelConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.hf_id, revision=cfg.revision, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_id, revision=cfg.revision, dtype=dtype
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def apply_lora(model, lora: LoraConfig, seed: int | None = None):
    """Wrap `model` with a PEFT LoRA adapter (expert projections via target_parameters)."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    if seed is not None:
        seed_everything(seed)  # LoRA A-matrix init draws from the global RNG
    kwargs = {}
    if lora.target_parameters:
        kwargs["target_parameters"] = list(lora.target_parameters)
    peft_cfg = PeftLoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules) if lora.target_modules else [],
        task_type="CAUSAL_LM",
        **kwargs,
    )
    return get_peft_model(model, peft_cfg)


def build(cfg: ModelConfig, lora: LoraConfig):
    """Return (model, tokenizer); LoRA-wrapped when enabled."""
    seed_everything(cfg.seed)
    if cfg.kind == "tiny-random":
        tokenizer = build_tokenizer()
        model = build_tiny_model(cfg, tokenizer)
    else:
        model, tokenizer = build_pretrained(cfg)
    if lora.enabled:
        model = apply_lora(model, lora, cfg.seed)
    return model, tokenizer


def param_dtypes(model) -> list[str]:
    """Sorted distinct parameter dtypes actually present (e.g. LoRA may be fp32 on a bf16 base)."""
    return sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters()})


def trainable_state(model) -> dict:
    """Clone of every trainable tensor (all weights for the tiny model, LoRA for PEFT)."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k in names}


def load_trainable_state(model, state: dict) -> None:
    import torch

    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v)


def reset_requires_grad(model) -> None:
    """Full fine-tune path: everything trainable. (PEFT models manage their own flags.)"""
    if not hasattr(model, "peft_config"):
        for p in model.parameters():
            p.requires_grad_(True)


def routing_snapshot(model, tokenizer, texts: list[str], max_tokens: int):
    """Router logits per layer for fixed probe texts: list[layer][token][expert]."""
    import torch

    from ptr_training.moe.routing_metrics import router_logits_to_lists

    was_training = model.training
    model.eval()
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_router_logits=True, use_cache=False)
    layers = router_logits_to_lists(out.router_logits, enc["attention_mask"])
    model.train(was_training)
    return [layer[:max_tokens] for layer in layers]
