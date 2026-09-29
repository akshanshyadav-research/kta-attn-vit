"""Notebook-derived ViT implementation. Batch size one; configured by build_model."""
import torch
from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer
from .ranking import kendall_scores


def pearson_corr(cls_token, tokens):
    # Original name retained for the revision analysis scripts.
    return kendall_scores(cls_token, tokens, variant="notebook")

class CustomAttention(nn.Module):

    def __init__(self, attn, num_heads, ats_ratio):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        self.ats_ratio = ats_ratio
        self.scale = attn.scale
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop

    def forward(self, x):
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        mask_attention = None
        avg_attn = attn.mean(dim=1)
        cls_token_attn = avg_attn[:, 0, :]
        threshold = torch.quantile(cls_token_attn, self.ats_ratio, dim=-1, keepdim=True)
        mask_attention = cls_token_attn >= threshold
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return (x, mask_attention)

class PatchDropout(nn.Module):

    def __init__(self, keep_ratio, variant):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.variant = variant

    def forward(self, x, mask):
        (batch_size, seq_len, embedding_dim) = x.shape
        assert seq_len == 197, f'Expected sequence length of 197, but got {seq_len}'
        cls_token = x[:, 0, :]
        tokens = x[:, 1:, :]
        distances = kendall_scores(cls_token, tokens, self.variant)
        k = int((seq_len - 1) * self.keep_ratio)
        (threshold_values, _) = torch.kthvalue(distances, min(k + 1, seq_len - 1), dim=1, keepdim=True)
        keep_mask = distances <= threshold_values
        keep_mask = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=x.device), keep_mask], dim=1)
        x_filtered = x[keep_mask].view(batch_size, -1, embedding_dim)
        return x_filtered

class CustomBlock(nn.Module):

    def __init__(self, block, ats_ratio):
        super(CustomBlock, self).__init__()
        self.norm1 = block.norm1
        self.attn = CustomAttention(block.attn, block.attn.num_heads, ats_ratio)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2

    def forward(self, x, apply_patch_drop=False):
        x_residual = self.norm1(x)
        (x_residual, mask_attention) = self.attn(x_residual)
        x = x + self.drop_path1(self.ls1(x_residual))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if apply_patch_drop:
            cls_token = x[:, 0].unsqueeze(1)
            other_tokens = x[:, 1:]
            selected_tokens = other_tokens[mask_attention[:, 1:]]
            selected_tokens = selected_tokens.unsqueeze(0)
            x = torch.cat((cls_token, selected_tokens), dim=1)
        return x

class CustomVisionTransformer(VisionTransformer):

    def __init__(self, patch_drop_blocks=None, keep_ratio=.85, ats_ratio=.05, variant="notebook", **kwargs):
        super(CustomVisionTransformer, self).__init__(**kwargs)
        self.kta_block = -1
        self.patch_drop_blocks = patch_drop_blocks if patch_drop_blocks is not None else []
        self.patch_drop = PatchDropout(keep_ratio, variant)
        self.blocks = nn.Sequential(*[CustomBlock(block, ats_ratio) if i in self.patch_drop_blocks else block for (i, block) in enumerate(self.blocks)])

    def forward(self, x, mask=None):
        B = x.shape[0]
        if B != 1:
            raise ValueError("Variable token pruning supports batch size one")
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if mask is not None and self.kta_block == -1:
            x = self.patch_drop(x, mask)
        for (i, block) in enumerate(self.blocks):
            if isinstance(block, CustomBlock):
                x = block(x, apply_patch_drop=True)
            else:
                x = block(x)
            if mask is not None and i == self.kta_block:
                x = self.patch_drop(x, mask)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


def build_model(model_name='vit_base_patch16_224', keep_ratio=0.85,
                ats_ratio=0.05, blocks=(3,4,5,6,7,8), variant='notebook', pretrained=True):
    if not 0 <= keep_ratio <= 1 or not 0 <= ats_ratio < 1:
        raise ValueError('keep_ratio must be in [0,1]; ats_ratio in [0,1)')
    if variant not in ('notebook', 'tau-b'):
        raise ValueError('Unknown Kendall variant')
    reference = timm.create_model(model_name, pretrained=pretrained)
    if any(b < 0 or b >= len(reference.blocks) for b in blocks):
        raise ValueError('ATS block outside model depth')
    model = CustomVisionTransformer(img_size=224, patch_size=16,
        embed_dim=reference.embed_dim, depth=len(reference.blocks),
        num_heads=reference.blocks[0].attn.num_heads, mlp_ratio=4,
        qkv_bias=True, norm_layer=nn.LayerNorm, num_classes=1000,
        patch_drop_blocks=list(blocks), keep_ratio=keep_ratio, ats_ratio=ats_ratio, variant=variant)
    model.load_state_dict(reference.state_dict(), strict=True)
    model.pretrained_cfg = reference.pretrained_cfg
    return model.eval()
