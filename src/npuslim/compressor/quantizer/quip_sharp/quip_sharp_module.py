"""
QuIP# Module - Per-layer quantization with E8 codebooks.

This module implements the core quantization logic for a single layer,
including:
1. Incoherence preprocessing (RHT)
2. LDLQ with codebook quantization
3. Fine-tuning (optional)
"""

import time
import copy
import torch
import torch.nn as nn
import transformers
from loguru import logger

from npuslim.compressor.core.base_hessian_module import BaseHessianModule
from npuslim.utils.backend import bh

from .codebook import get_codebook
from .utils.hadamard import (
    get_hadK,
    RHT_H,
    RHT_W,
    matmul_hadU_cuda,
)


class QuIPSharpModule(BaseHessianModule):
    """
    QuIP# Module for per-layer quantization.

    Extends BaseHessianModule to add:
    - E8 codebook-based vector quantization
    - Randomized Hadamard Transform for incoherence
    - LDLQ algorithm adapted for codebooks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # QuIP# specific config
        self.w_bits = getattr(self.config, "w_bits", 2)
        self.codebook_name = getattr(self.config, "codebook", "E8P12")
        self.incoh_mode = getattr(self.config, "incoh_mode", "had")
        self.rescale_WH = getattr(self.config, "rescale_WH", True)
        self.scale_override = getattr(self.config, "scale_override", -1.0)
        self.resid_scale_override = getattr(self.config, "resid_scale_override", -1.0)
        self.quip_tune_iters = getattr(self.config, "quip_tune_iters", 10)
        self.blocksize = getattr(self.config, "blocksize", 128)
        self.use_fp64 = getattr(self.config, "use_fp64", False)
        self.fake_quant = getattr(self.config, "fake_quant", False)
        self.lora_rank = getattr(self.config, "lora_rank", 0)
        self.sigma_reg = getattr(self.config, "sigma_reg", 1e-2)
        self.sigma_reg2 = getattr(self.config, "sigma_reg2", 1e-2)

        # Initialize codebook
        self.codebook = get_codebook(self.codebook_name, inference=False)
        self.codesz = self.codebook.codesz

        # Validate dimensions
        if self.columns % self.codesz != 0:
            logger.warning(
                f"⚠️ Layer {self.name}: in_features ({self.columns}) not divisible by "
                f"code size ({self.codesz}). Skipping this layer."
            )
            self._skip_layer = True
        else:
            self._skip_layer = False

        # Preprocessing parameters
        self.SU = None  # Input dimension signs
        self.SV = None  # Output dimension signs
        self.scaleWH = None  # Diagonal rescaling
        self.hadK = None  # Hadamard matrix for non-power-of-2

    def fasterquant(self, **kwargs):
        """
        Execute QuIP# quantization for this layer.

        Workflow:
        1. Get weight and Hessian
        2. Incoherence preprocessing (RHT)
        3. LDLQ with codebook quantization
        4. (Optional) Fine-tuning
        5. Incoherence postprocessing
        6. Pack results

        Returns:
            dict with quantization parameters
        """
        if self._skip_layer:
            logger.warning(f"⚠️ Skipping layer {self.name} (dimension mismatch)")
            return None

        # Get weight
        w = self.layer.weight.data.clone().float()
        if isinstance(self.layer, nn.Conv2d):
            w = w.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            w = w.t()

        full_w = w.clone()
        tick = time.time()

        # Get Hessian
        H = self.H.data.clone()

        # Move codebook to same device as weights
        device = w.device
        self.codebook = self.codebook.to(device)

        # Regularize Hessian
        H = self._regularize_hessian(H)

        # Numerical precision
        dtype_ = torch.float64 if self.use_fp64 else torch.float32
        w = w.to(dtype_)
        H = H.to(dtype_)

        # === Incoherence Preprocessing ===
        Lhr, Hr, Wr, SU, SV, scaleWH = self._incoherence_preprocess(H, w)

        if Lhr is None:
            # Fallback to fp64
            logger.warning(f"⚠️ Layer {self.name}: Cholesky failed, retrying with fp64")
            dtype_ = torch.float64
            H = H.to(dtype_)
            w = full_w.to(dtype_)
            Lhr, Hr, Wr, SU, SV, scaleWH = self._incoherence_preprocess(H, w)
            if Lhr is None:
                raise RuntimeError(f"Layer {self.name}: Cholesky decomposition failed even with fp64")

        # === LDLQ Quantization ===
        Wo = Wr.clone()  # Save for low-rank correction

        # Scale weight for codebook
        Wscale = Wr.square().mean().sqrt()
        if self.scale_override > 0:
            Wscale = Wscale / self.scale_override
        else:
            Wscale = Wscale / self.codebook.opt_scale

        Wr = Wr / Wscale

        # Block LDL decomposition
        L, D = self._block_ldl(Hr)

        # LDLQ with codebook
        hatWr, Qidxs = self._ldlq_buffered(Wr, Hr, L, D)

        hatWr = hatWr * Wscale

        # === (Optional) Low-rank Correction ===
        A, B = None, None
        if self.lora_rank > 0:
            hatWr, A, B = self._low_rank_process(Wo, hatWr, Lhr)

        # === Incoherence Postprocessing ===
        hatW = self._incoherence_postprocess(hatWr, SU, SV, scaleWH)

        # Pack indices
        Qidxs_packed = self.codebook.maybe_pack_idxs(Qidxs)

        # Update layer weight (for fake quant)
        if isinstance(self.layer, transformers.Conv1D):
            hatW_for_layer = hatW.t()
        else:
            hatW_for_layer = hatW

        self.layer.weight.data = hatW_for_layer.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        # Compute error metrics
        quant_w_preproc = hatW.clone()
        if isinstance(self.layer, nn.Conv2d):
            quant_w_preproc = quant_w_preproc.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            quant_w_preproc = quant_w_preproc.t()

        norm_loss = torch.norm(quant_w_preproc - full_w).item()
        self.error = (
            ((full_w - quant_w_preproc) @ H.type(torch.float) @ (full_w - quant_w_preproc).T)
            .trace()
            .item()
        )
        self.Hmag = H.max().item()

        # Timing
        self.time = time.time() - tick
        self.print_log(avg_loss=self.error / self.nsamples, norm_loss=norm_loss)

        # Collect results
        result = {
            "Qidxs": Qidxs_packed.cpu(),
            "SU": SU.to(torch.float16).cpu(),
            "SV": (SV * Wscale.to(SV.device)).to(torch.float16).cpu(),  # Fuse Wscale into SV
            "scaleWH": scaleWH.cpu() if scaleWH is not None else None,
            "A": A.half().cpu() if A is not None else None,
            "B": B.half().cpu() if B is not None else None,
            "bias": self.layer.bias.data.cpu() if self.layer.bias is not None else None,
        }

        self.free()
        return result

    def _regularize_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """Regularize Hessian diagonal."""
        H.diagonal().add_(self.sigma_reg * H.diag().mean())
        return H

    def _incoherence_preprocess(self, H: torch.Tensor, W: torch.Tensor):
        """
        Apply incoherence preprocessing.

        Returns:
            Lhr: Cholesky factor of transformed Hessian
            Hr: Transformed Hessian
            Wr: Transformed weight
            SU: Input dimension signs
            SV: Output dimension signs
            scaleWH: Diagonal rescaling (optional)
        """
        dtype_ = H.dtype
        device = H.device
        m, n = W.shape

        scaleWH = None
        Wr = W.clone()
        Hr = H.clone()

        # Diagonal rescaling
        if self.rescale_WH:
            Hr = Hr / Hr.abs().max()
            diagH = torch.diag(Hr)
            diagW2 = torch.diag(Wr.T @ Wr)
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
            scaleWH = scaleWH.clamp(min=1e-8)
            Wr = Wr * scaleWH[None, :]
            Hr = Hr / scaleWH[None, :]
            Hr = Hr / scaleWH[:, None]
            scaleWH = scaleWH.cpu()

        # Randomized Hadamard Transform
        if self.incoh_mode == "had":
            # Generate random signs
            SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
            SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)

            # Get Hadamard matrices for both dimensions
            hadK_n, K_n = get_hadK(n)  # For input dimension
            hadK_m, K_m = get_hadK(m)  # For output dimension

            # Transform H (only uses n dimension)
            Hr = RHT_H(Hr, SU, hadK_n)

            # Transform W (uses both m and n dimensions)
            Wr = RHT_W(Wr, SU, SV, hadK_n, hadK_m)

            self.hadK = hadK_n
            self.hadK_m = hadK_m
        else:
            raise NotImplementedError(f"incoh_mode={self.incoh_mode} not implemented. Use 'had'.")

        SV = SV.cpu()
        SU = SU.cpu()

        # Cholesky decomposition
        try:
            Lhr = torch.linalg.cholesky(Hr)
            if not torch.all(torch.isfinite(Lhr)):
                return None, None, None, None, None, None
        except RuntimeError:
            return None, None, None, None, None, None

        return Lhr, Hr, Wr.to(device), SU, SV, scaleWH

    def _incoherence_postprocess(
        self,
        hatWr: torch.Tensor,
        SU: torch.Tensor,
        SV: torch.Tensor,
        scaleWH: torch.Tensor,
    ) -> torch.Tensor:
        """Reverse incoherence preprocessing."""
        device = hatWr.device

        if self.incoh_mode == "had":
            # Use the same Hadamard matrices from preprocessing (stored in self)
            hadK_n = self.hadK
            hadK_m = self.hadK_m
            # Forward RHT_W (from original quip-sharp):
            #   RHT_W(W, SU, SV) = matmul_hadUt(matmul_hadUt(W.T * SV).T * SU)
            #   = had @ ((had @ (W.T * SV)).T * SU)
            #
            # Reverse (incoherence_process from original):
            #   hatWr = (matmul_hadU((matmul_hadU(hatWr) * SU).T) * SV).T
            #
            # Our implementation with explicit hadK:
            # Step 1: Apply H_n to hatWr [m, n] → [m, n]
            # Step 2: Divide by SU [m, n] / [1, n] → [m, n]
            # Step 3: Transpose → [n, m]
            # Step 4: Apply H_m → [n, m]
            # Step 5: Divide by SV [n, m] / [1, m] → [n, m]
            # Step 6: Transpose back → [m, n]
            hatWr = matmul_hadU_cuda(hatWr, hadK_n)  # [m, n]
            hatWr = hatWr / SU.to(device).unsqueeze(0)  # [m, n] / [1, n]
            hatWr = matmul_hadU_cuda(hatWr.T, hadK_m)  # [n, m], no transpose yet!
            hatWr = hatWr / SV.to(device).unsqueeze(0)  # [n, m] / [1, m]
            hatWr = hatWr.T  # [m, n]

        # Reverse diagonal rescaling
        if scaleWH is not None:
            hatWr = hatWr / scaleWH[None, :].to(device)

        assert torch.isfinite(hatWr).all()
        return hatWr

    def _block_ldl(self, H: torch.Tensor):
        """Block LDL decomposition."""
        codesz = self.codesz
        n = H.shape[0]

        # Cholesky
        L = torch.linalg.cholesky(H)
        D = L.diag() ** 2

        # Normalize L
        L = L @ torch.diag(1 / L.diag())
        L = L - torch.eye(n, device=L.device, dtype=L.dtype)

        return L, D

    def _ldlq_buffered(
        self,
        Wr: torch.Tensor,
        Hr: torch.Tensor,
        L: torch.Tensor,
        D: torch.Tensor,
    ) -> tuple:
        """
        LDLQ with codebook quantization (buffered version).

        Processes weights in reverse order, quantizing groups of `codesz` elements
        using the codebook.

        Based on: https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/algo/quip.py
        """
        codesz = self.codesz
        m, n = Wr.shape
        buf_cols = self.blocksize

        assert n % codesz == 0
        assert buf_cols % codesz == 0

        n_codes = n // codesz
        buf_size = buf_cols // codesz

        # Initialize outputs (using transposed for efficiency, matching original)
        hatWr_T = torch.zeros(n, m, dtype=Wr.dtype, device=Wr.device)
        Qidxs_T = torch.zeros(n_codes, m, dtype=self.codebook.idx_dtype, device=Wr.device)

        # Move Wr to CPU and work with transposed version
        device = Wr.device
        Wr_T = Wr.T.contiguous().to(device)
        Hr_T = Hr.T.contiguous().to(device)

        # Product cache
        prod_cache = torch.zeros(n, m, dtype=Wr.dtype, device=device)

        # === First pass: LDLQ quantization ===
        for cur_col in range(n_codes, 0, -buf_size):
            b_start = (cur_col - buf_size) * codesz
            b_end = cur_col * codesz

            b_Wr_T = Wr_T[b_start:b_end]
            b_hatWr_T = hatWr_T[b_start:b_end]
            b_L = L[b_start:b_end].contiguous()
            b_prod = prod_cache[b_start:b_end]
            b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]

            L_offset = b_start

            for i in reversed(range(buf_size)):
                col_start = i * codesz
                col_end = (i + 1) * codesz

                # WXWX = Wr + (Wr - hatWr) @ L (for columns after current)
                WXWX = b_Wr_T[col_start:col_end] + b_prod[col_start:col_end]

                if col_end < (b_end - b_start):
                    WXWX = WXWX + b_L[col_end:, col_start:col_end].T @ (
                        b_Wr_T[col_end:] - b_hatWr_T[col_end:]
                    )

                # Quantize with codebook
                q_vals, q_idx = self.codebook.quantize(
                    WXWX.T,  # Transpose back to [m, codesz]
                    resid_scale_override=self.resid_scale_override,
                )
                b_hatWr_T[col_start:col_end] = q_vals.T  # Store transposed
                b_Qidxs_T[i] = q_idx

            # Update cache for all columns (accumulates across entire matrix)
            prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)

        # === Second pass: Iterative refinement ===
        for ie in range(self.quip_tune_iters):
            delta_T = Wr_T - hatWr_T

            for cur_col in range(n_codes, 0, -buf_size):
                b_start = (cur_col - buf_size) * codesz
                b_end = cur_col * codesz

                b_hatWr_T = hatWr_T[b_start:b_end]
                b_Hr_T = Hr_T[b_start:b_end]
                b_delta_T = delta_T[b_start:b_end]
                b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]

                Hr_offset = b_start

                for i in reversed(range(buf_size)):
                    col_start = i * codesz
                    col_end = (i + 1) * codesz

                    # Compute refinement
                    Hr_block = b_Hr_T[col_start:col_end, col_start:col_end]
                    if codesz > 1:
                        Hr_inv = torch.linalg.inv(Hr_block.T).T
                    else:
                        Hr_inv = 1 / b_Hr_T[i:i+1, col_start:col_end]

                    # Correct order: Hr_inv @ Hr_block @ delta_T
                    WXWX = b_hatWr_T[col_start:col_end] + Hr_inv @ b_Hr_T[col_start:col_end] @ delta_T

                    b_delta_T[col_start:col_end] += b_hatWr_T[col_start:col_end]

                    # Re-quantize
                    if ie < self.quip_tune_iters - 1:
                        q_vals, _ = self.codebook.quantize(
                            WXWX.T,
                            resid_scale_override=self.resid_scale_override,
                        )
                        b_hatWr_T[col_start:col_end] = q_vals.T
                    else:
                        q_vals, q_idx = self.codebook.quantize(
                            WXWX.T,
                            resid_scale_override=self.resid_scale_override,
                        )
                        b_hatWr_T[col_start:col_end] = q_vals.T
                        b_Qidxs_T[i] = q_idx

                    b_delta_T[col_start:col_end] -= b_hatWr_T[col_start:col_end]

        # Transpose back
        hatWr = hatWr_T.T.contiguous()
        Qidxs = Qidxs_T.T.contiguous()

        return hatWr, Qidxs

    def _low_rank_process(
        self,
        Wo: torch.Tensor,
        hatWr: torch.Tensor,
        Lhr: torch.Tensor,
    ) -> tuple:
        """Apply low-rank correction (LoRA-style)."""
        # SVD of residual
        residual = (Wo - hatWr) @ Lhr
        svdRZ = torch.linalg.svd(residual, full_matrices=False)

        A = svdRZ.U[:, :self.lora_rank]
        B = torch.linalg.solve_triangular(
            Lhr,
            torch.diag(svdRZ.S[:self.lora_rank]) @ svdRZ.Vh[:self.lora_rank],
            upper=False,
            left=False,
        )

        # Balance A and B
        svdB = torch.linalg.svd(B, full_matrices=False)
        A = (A @ svdB.U @ torch.diag(svdB.S.sqrt())).half()
        B = (torch.diag(svdB.S.sqrt()) @ svdB.Vh).half()

        # Apply correction
        hatWr = hatWr.to(A.device) + (A @ B).to(hatWr.dtype)

        return hatWr, A, B

    def print_log(self, avg_loss, norm_loss):
        """Print quantization statistics."""
        w_shape = self.layer.weight.shape
        w_out, w_in = w_shape[0], w_shape[1]
        label_width = 25
        logger_info = [
            f"{'Layer Shape:':<{label_width}} [Out={w_out}, In={w_in}]",
            f"{'Hessian Error:':<{label_width}} {avg_loss:.6f}",
            f"{'Norm Loss (L2):':<{label_width}} {norm_loss:.6f}",
            f"{'Hessian Max:':<{label_width}} {self.Hmag:.6f}",
            f"{'Time:':<{label_width}} {self.time:.2f}s",
        ]
        for line in logger_info:
            logger.info(line)
