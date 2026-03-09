"""
QuIP# Configuration.

QuIP# supports 2-bit, 3-bit, and 4-bit quantization using E8 lattice codebooks.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List


@dataclass
class QuIPSharpConfig:
    """
    QuIP# (QuIP-Sharp) algorithm specific configuration.

    QuIP# uses:
    1. Randomized Hadamard Transform for incoherence processing
    2. E8 lattice codebooks for vector quantization
    3. Optional fine-tuning for improved fidelity
    """

    # === Bit-width and Codebook ===
    w_bits: int = 2  # 2, 3, or 4 bits
    codebook: str = "E8P12"  # E8P12 (2-bit), E8P12RVQ3B (3-bit), E8P12RVQ4B (4-bit)

    # === Incoherence Processing ===
    incoh_mode: Literal["had", "kron"] = "had"  # "had" (Hadamard) or "kron" (Kronecker/butterfly)
    rescale_WH: bool = True  # Diagonal rescaling of W and H

    # === LDLQ Parameters ===
    scale_override: float = -1.0  # Manual scale override (<0 = auto from codebook.opt_scale)
    resid_scale_override: float = -1.0  # Residual scale override for RVQ
    quip_tune_iters: int = 10  # Iterative refinement passes
    blocksize: int = 128  # Buffer size for LDLQ

    # === Low-rank Correction (Optional) ===
    lora_rank: int = 0  # LoRA rank (0 = disabled)
    sigma_reg: float = 1e-2  # Hessian regularization
    sigma_reg2: float = 1e-2  # Low-rank regularization

    # === Fine-tuning (Optional) ===
    ft_epochs: int = 0  # Fine-tuning epochs (0 = disabled)
    ft_lr: float = 5e-5  # Fine-tuning learning rate
    ft_susv_lr: float = 5e-4  # Higher LR for SU/SV
    ft_batch_size: int = 4  # Fine-tuning batch size
    ft_update_freq: int = 2  # Gradient accumulation steps
    ft_valid_freq: int = 1  # Validation frequency
    ft_early_stop: int = 3  # Early stopping patience
    ft_train_mode: bool = False  # Manifest weights during training
    ft_grad_ckpt: bool = False  # Gradient checkpointing

    # === Numerical Precision ===
    use_fp64: bool = False  # Use float64 for numerical stability

    # === Inference Mode ===
    fake_quant: bool = False  # Skip packing (for testing)

    # === Layer Selection ===
    ignore_layers: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and adjust configuration."""
        # Validate bit-width and codebook match
        valid_combinations = {
            2: ["E8P12"],
            3: ["E8P12RVQ3B"],
            4: ["E8P12RVQ4B"],
        }

        if self.w_bits not in valid_combinations:
            raise ValueError(f"w_bits must be 2, 3, or 4, got {self.w_bits}")

        # Check if codebook is implemented
        available_codebooks = ["E8P12", "E8P12RVQ4B"]
        if self.codebook not in available_codebooks:
            raise NotImplementedError(
                f"Codebook {self.codebook} not yet implemented. "
                f"Available: {available_codebooks}"
            )

        # Set default scale from codebook if not overridden
        if self.scale_override < 0:
            # Will be set from codebook.opt_scale during quantization
            pass
