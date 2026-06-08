from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def forward(
        self,
        model_output: torch.Tensor,
        ut: torch.Tensor,
        description: str = "",
    ) -> dict[str, torch.Tensor]:
        mse = F.mse_loss(model_output.float(), ut.float())
        loss = torch.nan_to_num(mse)
        prefix = description if description else "Flow"
        return {f"{prefix}/loss": loss, f"{prefix}/mse": loss}

