"""Training-preset switching helpers for handler decomposition."""

from typing import Tuple


class TrainingPresetMixin:
    """Helpers for switching runtime initialization to training-safe settings."""

    def switch_to_training_preset(self) -> Tuple[str, bool]:
        """Reinitialize with quantization disabled using the last init parameters."""
        if self.quantization is None:
            return "Already in training-safe preset (quantization disabled).", True

        if not self.last_init_params:
            return "Cannot switch preset automatically: no previous init parameters found.", False

        params = dict(self.last_init_params)
        params["quantization"] = None

        status, ok = self.initialize_service(
            project_root=params["project_root"],
            config_path=params["config_path"],
            device=params["device"],
            use_flash_attention=params["use_flash_attention"],
            compile_model=params["compile_model"],
            offload_to_cpu=params["offload_to_cpu"],
            offload_dit_to_cpu=params["offload_dit_to_cpu"],
            quantization=None,
            prefer_source=params.get("prefer_source"),
        )
        if ok:
            return f"Switched to training preset (quantization disabled).\n{status}", True
        return f"Failed to switch to training preset.\n{status}", False
