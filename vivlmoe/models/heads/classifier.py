"""ViMACSA multi-image aspect-sentiment classifier (paper Sec. 6.1, Fig. 4c; plan decision 18).

Batch: ``pixel_values [B,M,3,H,W]``, ``image_mask [B,M]`` bool, ``input_ids/attention_mask [B,A,L]``
(one text segment per aspect, legacy protocol), ``labels [B,A]``. Images are encoded once per sample
(valid slots only), replicated per aspect and concatenated with the aspect text for a joint pass with
the concatenated key mask. Loss = sum over aspects of the batch-mean CE (legacy scaling).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vivlmoe.config.schema import ModelCfg
from vivlmoe.models.backbone import ViVLBackbone
from vivlmoe.models.outputs import ClassifierOutput
from vivlmoe.models.projections import init_module_


class MultiImageClassifier(nn.Module):
    def __init__(self, backbone: ViVLBackbone, cfg: ModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        head = cfg.classifier
        in_dim = 2 * backbone.dim if head.pool == "concat_mean_v_t" else backbone.dim
        self.cls_head = nn.Sequential(nn.Dropout(head.dropout), nn.Linear(in_dim, head.num_labels))
        init_module_(self.cls_head, cfg.init_std)

    def _encode_images(
        self, pixel_values: torch.Tensor, image_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B,M,3,H,W]`` -> tokens ``[B, M*N, d]`` (zeros in padded slots) and mask ``[B, M*N]``."""
        b, m = pixel_values.shape[:2]
        n = self.backbone.num_visual_tokens
        flat_mask = image_mask.reshape(b * m).bool()
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            raise ValueError("classifier batch contains no valid image (image_mask all False)")
        encoded = self.backbone.encode_image_tokens(pixel_values.flatten(0, 1)[valid_idx])
        tokens = torch.zeros(
            b * m, n, self.backbone.dim, dtype=encoded.dtype, device=encoded.device
        )
        tokens = tokens.index_put((valid_idx,), encoded)
        token_mask = flat_mask.view(b, m, 1).expand(b, m, n).reshape(b, m * n)
        return tokens.view(b, m * n, -1), token_mask

    def forward(self, batch: dict, mode: str = "joint_cls") -> ClassifierOutput:
        if mode != "joint_cls":
            raise ValueError(f"MultiImageClassifier: unknown mode {mode!r}")
        px, image_mask = batch["pixel_values"], batch["image_mask"]
        ids, attn = batch["input_ids"], batch["attention_mask"]
        if px.ndim != 5 or ids.ndim != 3:
            raise ValueError(
                f"expected pixel_values [B,M,3,H,W] and input_ids [B,A,L], got {tuple(px.shape)} / {tuple(ids.shape)}"
            )
        b, a, length = ids.shape
        img_tokens, img_mask = self._encode_images(px, image_mask)
        n_img = img_tokens.shape[1]
        img_tokens = img_tokens.unsqueeze(1).expand(b, a, n_img, -1).reshape(b * a, n_img, -1)
        img_mask = img_mask.unsqueeze(1).expand(b, a, n_img).reshape(b * a, n_img)
        text_mask = attn.reshape(b * a, length).bool()
        text_tokens = self.backbone.encode_text_tokens(
            ids.reshape(b * a, length), attn.reshape(b * a, length)
        )
        enc, key_mask, _ = self.backbone.forward_joint(img_tokens, img_mask, text_tokens, text_mask)
        h = enc.last_hidden_state
        pool = self.cfg.classifier.pool
        if pool == "mean_joint":
            pooled = self.backbone.masked_mean(h, key_mask)
        elif pool == "mean_text":
            pooled = self.backbone.masked_mean(h[:, n_img:], text_mask)
        else:  # concat_mean_v_t (legacy parity); a zero-image sample gets a zero visual vector
            pooled = torch.cat(
                [
                    self.backbone.masked_mean(h[:, :n_img], img_mask),
                    self.backbone.masked_mean(h[:, n_img:], text_mask),
                ],
                dim=-1,
            )
        logits = self.cls_head(pooled.to(h.dtype)).float().view(b, a, -1)
        loss = None
        labels = batch.get("labels")
        if labels is not None:
            loss = sum(F.cross_entropy(logits[:, i], labels[:, i]) for i in range(a))
        return ClassifierOutput(logits=logits, loss=loss, router_outputs=enc.router_outputs)
