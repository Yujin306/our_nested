import math
from einops import rearrange
import torch.nn as nn
import torch
from .layers import get_config, Mlp

# ---------------- MSG Gate ----------------
class MSGGate(nn.Module):
    def __init__(self, hidden_size, model_num):
        super().__init__()
        self.model_num = model_num
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, modality_embeddings):  # List of [B, N, C]
        weights = []
        for emb in modality_embeddings:
            w = self.gate(emb.mean(dim=1))  # [B, 1]
            weights.append(w)
        weights = torch.stack(weights, dim=1)  # [B, M, 1]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # normalize

        weighted = [emb * weights[:, i:i+1, :] for i, emb in enumerate(modality_embeddings)]
        return torch.cat(weighted, dim=1)  # [B, N*M, C]

# ---------------- Embedding ----------------
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_size = config.img_size
        patch_size = config.patch_size
        in_channels = config.in_channels
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.patch_embeddings = nn.Conv3d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)  # [B, C, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.position_embeddings
        return self.dropout(x)

# ---------------- Cross-Attention ----------------
class AttentionCrossModal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.head_dim * self.num_heads

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B, H, N, D]

    def forward(self, q, kv):
        q = self.transpose_for_scores(self.query(q))
        k = self.transpose_for_scores(self.key(kv))
        v = self.transpose_for_scores(self.value(kv))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs = self.softmax(scores)
        probs = self.attn_dropout(probs)

        context = torch.matmul(probs, v)  # [B, H, N, D]
        context = context.permute(0, 2, 1, 3).contiguous().view(q.size(0), -1, self.all_head_size)
        out = self.out(context)
        return self.proj_dropout(out)

# ---------------- FeedForward + Attention ----------------
class CrossAttBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.attn = AttentionCrossModal(config)
        self.ffn = Mlp(config)

    def forward(self, q, kv):
        x = self.attn(q, kv) + q
        x = self.attn_norm(x)
        x = self.ffn(x) + x
        x = self.ffn_norm(x)
        return x

# ---------------- Token Learner ----------------
class TokenLearner(nn.Module):
    def __init__(self, in_channels, S):
        super().__init__()
        self.token_conv = nn.Conv3d(in_channels, S, kernel_size=3, padding=1)

    def forward(self, x):
        selected = torch.sigmoid(self.token_conv(x))  # [B, S, D, H, W]
        selected = rearrange(selected, "b s d h w -> b s (d h w) 1")
        x = rearrange(x, "b c d h w -> b 1 (d h w) c")
        return (x * selected).mean(dim=2)  # [B, S, C] → mean over spatial

# ---------------- Cross-Modality Fusion with MSG ----------------
class CrossModalityFusion(nn.Module):
    def __init__(self, model_num, in_channels, hidden_size, img_size,
                 mlp_size=256, token_mixer_size=32, token_learner=False, use_msg=True):
        super().__init__()
        patch_size = (1, 1, 1)
        self.config = get_config(in_channels=in_channels, hidden_size=hidden_size,
                                 patch_size=patch_size, img_size=img_size, mlp_dim=mlp_size)

        self.model_num = model_num
        self.img_size = img_size
        self.token_learner = token_learner
        self.use_msg = use_msg

        patch_num = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

        self.embeddings = nn.ModuleList([Embeddings(self.config) for _ in range(model_num)])

        if token_learner:
            self.token_mixer = TokenLearner(in_channels=in_channels, S=token_mixer_size)
        else:
            self.token_mixer = nn.Linear(patch_num, token_mixer_size)

        if use_msg:
            self.msg_gate = MSGGate(hidden_size, model_num)

        self.cross_attention = CrossAttBlock(self.config)

    def forward(self, q, kv):
        q = rearrange(q, "b c d h w -> b (d h w) c")  # [B, N, C]
        embed_list = []

        for i in range(self.model_num):
            x = self.embeddings[i](kv[:, i])  # [B, N, C]
            if self.token_learner:
                x = rearrange(x, "b (d h w) c -> b c d h w", d=self.img_size[0], h=self.img_size[1], w=self.img_size[2])
                x = self.token_mixer(x)
            else:
                x = x.transpose(1, 2)
                x = self.token_mixer(x)
                x = x.transpose(1, 2)
            embed_list.append(x)

        if self.use_msg:
            fused = self.msg_gate(embed_list)
        else:
            fused = torch.cat(embed_list, dim=1)

        x = self.cross_attention(q, fused)  # [B, N, C]
        x = x.transpose(1, 2)
        x = x.view(q.size(0), self.config.hidden_size, *self.img_size)
        return x

# ---------------- Test ----------------
if __name__ == "__main__":
    q = torch.rand(1, 64, 16, 16, 16)
    kv = torch.rand(1, 3, 64, 16, 16, 16)

    model = CrossModalityFusion(model_num=3,
                                in_channels=64,
                                hidden_size=64,
                                img_size=(16, 16, 16),
                                token_learner=True,
                                token_mixer_size=32,
                                use_msg=True)

    out = model(q, kv)
    print(out.shape)  # Expected: [1, 64, 16, 16, 16]
