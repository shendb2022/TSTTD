import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat

'''
This part for designing the network to obtain the discriminative features
'''


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, group_nums):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(1, depth + 1):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim)),
                nn.Conv2d(group_nums, group_nums, kernel_size=(1, 2), stride=1, padding=0)
            ]))

    def forward(self, x):
        for attn, ff, cov2D in self.layers:
            prex = torch.unsqueeze(x, dim=-1)
            x = attn(x) + x
            x = ff(x) + x
            x = torch.unsqueeze(x, dim=-1)
            union = torch.cat([prex, x], dim=-1)
            x = cov2D(union)
            x = torch.squeeze(x, dim=-1)
        return x


class SpectralGroupAttention(nn.Module):
    def __init__(self, band=189, m=20, d=128, depth=4, heads=4, dim_head=64, mlp_dim=64, adjust=False):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(m, d),
            nn.LeakyReLU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.pos_embedding = nn.Parameter(torch.randn(1, band + 1, d))
        self.transformer = Transformer(dim=d, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                       group_nums=band + 1)
        if adjust:
            self.adjust = nn.Sequential(
                nn.Linear(d, mlp_dim),
                nn.LeakyReLU(),
                nn.Linear(mlp_dim, mlp_dim // 2)
            )
        else:
            self.adjust = nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        class_token = x[:, 0]
        features = self.adjust(class_token)
        return features


if __name__ == '__main__':
    import scipy.io as sio
    from Tools import standard

    mat = sio.loadmat('/home/sdb/Codes/datasets2/Sandiego.mat')
    data = mat['data']
    data = standard(data)
    h, w, c = data.shape
    data = np.reshape(data, [-1, c], order='F')
    tp_sample = data[100:110]

    ### divide the spectrum into n overlapping groups
    m = 20
    pad_size = m // 2
    new_sample = np.pad(tp_sample, ((0, 0), (pad_size, pad_size)),
                        mode='symmetric')
    group_spectra = np.zeros([10, c, m])
    for i in range(c):
        group_spectra[:, i, :] = np.squeeze(new_sample[:, i:i + m])

    group_spectra = torch.from_numpy(group_spectra).float()
    model = SpectralGroupAttention(band=c, m=m, d=128)
    features = model(group_spectra)
    print(features.shape)
