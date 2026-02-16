"""Checkpoint sync and model-loading helpers for service initialization."""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger

from acestep.model_downloader import (
    check_main_model_exists,
    check_model_exists,
    ensure_dit_model,
    ensure_main_model,
)


class InitServiceLoaderMixin:
    """Helpers for model availability checks and component loading."""

    def _ensure_models_present(
        self,
        *,
        checkpoint_path: Path,
        config_path: str,
        prefer_source: Optional[str],
    ) -> Optional[Tuple[str, bool]]:
        """Ensure required checkpoint assets exist locally, downloading when missing."""
        if not check_main_model_exists(checkpoint_path):
            logger.info("[initialize_service] Main model not found, starting auto-download...")
            success, msg = ensure_main_model(checkpoint_path, prefer_source=prefer_source)
            if not success:
                return f"âŒ Failed to download main model: {msg}", False
            logger.info(f"[initialize_service] {msg}")

        if config_path == "":
            logger.warning(
                "[initialize_service] Empty config_path; pass None to use the default model."
            )

        if not check_model_exists(config_path, checkpoint_path):
            logger.info(f"[initialize_service] DiT model '{config_path}' not found, starting auto-download...")
            success, msg = ensure_dit_model(config_path, checkpoint_path, prefer_source=prefer_source)
            if not success:
                return f"âŒ Failed to download DiT model '{config_path}': {msg}", False
            logger.info(f"[initialize_service] {msg}")

        return None

    @staticmethod
    def _sync_model_code_if_needed(config_path: str, checkpoint_path: Path) -> None:
        """Sync model-side python files when checkpoint code metadata diverges."""
        from acestep.model_downloader import _check_code_mismatch, _sync_model_code_files

        mismatched = _check_code_mismatch(config_path, checkpoint_path)
        if mismatched:
            logger.warning(
                f"[initialize_service] Model code mismatch detected for '{config_path}': "
                f"{mismatched}. Auto-syncing from acestep/models/..."
            )
            _sync_model_code_files(config_path, checkpoint_path)
            logger.info("[initialize_service] Model code files synced successfully.")

    def _load_main_model_from_checkpoint(
        self,
        *,
        model_checkpoint_path: str,
        device: str,
        use_flash_attention: bool,
        compile_model: bool,
        quantization: Optional[str],
    ) -> str:
        """Load DiT, apply compile/quantization options, and return selected attention backend."""
        from transformers import AutoModel

        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"ACE-Step V1.5 checkpoint not found at {model_checkpoint_path}")

        if torch.cuda.is_available():
            if getattr(self, "model", None) is not None:
                del self.model
                self.model = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if use_flash_attention and self.is_flash_attention_available(device):
            attn_implementation = "flash_attention_2"
        else:
            if use_flash_attention:
                logger.warning(
                    f"[initialize_service] Flash attention requested but unavailable for device={device}. "
                    "Falling back to SDPA."
                )
            attn_implementation = "sdpa"

        attn_candidates = [attn_implementation]
        if "sdpa" not in attn_candidates:
            attn_candidates.append("sdpa")
        if "eager" not in attn_candidates:
            attn_candidates.append("eager")

        last_attn_error = None
        self.model = None
        for candidate in attn_candidates:
            try:
                logger.info(f"[initialize_service] Attempting to load model with attention implementation: {candidate}")
                self.model = AutoModel.from_pretrained(
                    model_checkpoint_path,
                    trust_remote_code=True,
                    attn_implementation=candidate,
                    torch_dtype=self.dtype,
                )
                attn_implementation = candidate
                break
            except Exception as exc:
                last_attn_error = exc
                logger.warning(f"[initialize_service] Failed to load model with {candidate}: {exc}")

        if self.model is None:
            raise RuntimeError(
                f"Failed to load model with attention implementations {attn_candidates}: {last_attn_error}"
            ) from last_attn_error

        self.model.config._attn_implementation = attn_implementation
        self.config = self.model.config

        if not self.offload_to_cpu:
            self.model = self.model.to(device).to(self.dtype)
        elif not self.offload_dit_to_cpu:
            logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
            self.model = self.model.to(device).to(self.dtype)
        else:
            self.model = self.model.to("cpu").to(self.dtype)
        self.model.eval()

        if compile_model:
            self._ensure_len_for_compile(self.model, "model")
            self.model = torch.compile(self.model)

            if quantization is not None:
                from torchao.quantization import quantize_
                from torchao.quantization.quant_api import _is_linear
                if quantization == "int8_weight_only":
                    from torchao.quantization import Int8WeightOnlyConfig
                    quant_config = Int8WeightOnlyConfig()
                elif quantization == "fp8_weight_only":
                    from torchao.quantization import Float8WeightOnlyConfig
                    quant_config = Float8WeightOnlyConfig()
                elif quantization == "w8a8_dynamic":
                    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, MappingType
                    quant_config = Int8DynamicActivationInt8WeightConfig(act_mapping_type=MappingType.ASYMMETRIC)
                else:
                    raise ValueError(f"Unsupported quantization type: {quantization}")

                def _dit_filter_fn(module, fqn):
                    """Keep only DiT linear layers and exclude tokenizer/detokenizer paths."""
                    if not _is_linear(module, fqn):
                        return False
                    for part in fqn.split("."):
                        if part in ("tokenizer", "detokenizer"):
                            return False
                    return True

                quantize_(self.model, quant_config, filter_fn=_dit_filter_fn)
                logger.info(f"[initialize_service] DiT quantized with: {quantization}")

        silence_latent_path = os.path.join(model_checkpoint_path, "silence_latent.pt")
        if not os.path.exists(silence_latent_path):
            raise FileNotFoundError(f"Silence latent not found at {silence_latent_path}")
        self.silence_latent = torch.load(silence_latent_path, weights_only=True).transpose(1, 2)
        self.silence_latent = self.silence_latent.to(device).to(self.dtype)
        return attn_implementation

    def _load_vae_model(self, *, checkpoint_dir: str, device: str, compile_model: bool) -> str:
        """Load and optionally compile the VAE module."""
        from diffusers.models import AutoencoderOobleck

        vae_checkpoint_path = os.path.join(checkpoint_dir, "vae")
        if not os.path.exists(vae_checkpoint_path):
            raise FileNotFoundError(f"VAE checkpoint not found at {vae_checkpoint_path}")

        self.vae = AutoencoderOobleck.from_pretrained(vae_checkpoint_path)
        if not self.offload_to_cpu:
            vae_dtype = self._get_vae_dtype(device)
            self.vae = self.vae.to(device).to(vae_dtype)
        else:
            vae_dtype = self._get_vae_dtype("cpu")
            self.vae = self.vae.to("cpu").to(vae_dtype)
        self.vae.eval()

        if compile_model:
            self._ensure_len_for_compile(self.vae, "vae")
            self.vae = torch.compile(self.vae)

        return vae_checkpoint_path

    def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir: str, device: str) -> str:
        """Load text tokenizer and embedding model."""
        from transformers import AutoModel, AutoTokenizer

        text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Text encoder not found at {text_encoder_path}")

        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
        if not self.offload_to_cpu:
            self.text_encoder = self.text_encoder.to(device).to(self.dtype)
        else:
            self.text_encoder = self.text_encoder.to("cpu").to(self.dtype)
        self.text_encoder.eval()
        return text_encoder_path
