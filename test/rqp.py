import math
import torch
import torch.nn as nn
import torch.autograd as autograd


class RQPGradCollector:
    """
    Reasoning-sensitive Fisher:
    F^(r) ≈ E_x,sum_(a,b) (∂(z_a - z_b)/∂W)^T (∂(z_a - z_b)/∂W)
    """

    def __init__(self, model, topk=5):
        self.model = model
        self.topk = topk
        self.F_r = {}
        self.nsamples = 0  # pair-sample 数
        self._setup_layers()

    def _setup_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.F_r[name] = torch.zeros(
                    (module.in_features, module.in_features),
                    device=module.weight.device,
                    dtype=torch.float32,
                )

    @torch.no_grad()
    def _select_topk_indices(self, logits):
        """
        logits: (N, V)
        return: (N, topk)
        """
        _, topi = torch.topk(logits, k=self.topk, dim=-1)
        return topi

    def collect_grads(self, input_ids):
        self.model.zero_grad(set_to_none=True)

        # ---- forward ----
        outputs = self.model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        logits = logits.view(-1, logits.size(-1)).float()  # (N, V)

        N = logits.size(0)
        topi = self._select_topk_indices(logits)

        # 当前 batch 的 pair 数（每个 sample 有 topk-1 个 pair）
        curr_pairs = N * (self.topk - 1)

        # ---- GPTQ-style: 先衰减旧统计 ----
        if self.nsamples > 0:
            for name in self.F_r:
                self.F_r[name] *= self.nsamples / (self.nsamples + curr_pairs)

        # ---- 对每个 pair 累积 ----
        for i in range(1, self.topk):
            z1 = logits[torch.arange(N), topi[:, 0]]
            zi = logits[torch.arange(N), topi[:, i]]
            gap = z1 - zi                     # (N,)

            # 🔑 核心修正：根据 gap 的维度自动缩放，类似 GPTQ 对 inp 的处理
            scale = math.sqrt(1.0 / (100 * curr_pairs))
            loss_scaled = (gap * scale).sum()

            grads = autograd.grad(
                loss_scaled,
                [m.weight for m in self._linear_modules()],
                retain_graph=(i < self.topk - 1),
                allow_unused=True,
            )

            for (name, module), g in zip(self._linear_named_modules(), grads):
                if g is None:
                    continue

                g = g.detach().float()        # (out, in)
                g_in = g.transpose(0, 1)      # (in, out)

                # 单 pair Fisher
                F_add = g_in @ g_in.transpose(0, 1)

                if not torch.isfinite(F_add).all():
                    continue

                # ---- GPTQ-style running average ----
                self.F_r[name] += F_add

        self.nsamples += curr_pairs
        self.model.zero_grad(set_to_none=True)

    def _linear_modules(self):
        return [m for m in self.model.modules() if isinstance(m, nn.Linear)]

    def _linear_named_modules(self):
        return [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear)]

    def get_merged_hessian(self, name, H_ppl, alpha=0.5, eps=1e-6):
        """
        H_RQP = alpha * H_PPL + (1 - alpha) * F_r
        """
        if name not in self.F_r:
            return H_ppl

        F_r = self.F_r[name]
        scale = H_ppl.norm() / (F_r.norm() + eps)
        F_r = F_r * scale

        return alpha * H_ppl + (1 - alpha) * F_r