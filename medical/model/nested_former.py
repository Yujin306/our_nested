import torch
import torch.nn as nn
from einops import rearrange
from .encoder.global_poolformer import Encoder, Convolution, TwoConv, UpCat
from .fusion.nmafa import NMaFaLayer


# ✅ MSG (Modality-Sensitive Gate)
class ModalityGate(nn.Module):
    def __init__(self, in_channels, model_num):
        super().__init__()
        total_channels = in_channels * model_num
        self.gate = nn.Sequential(
            nn.Conv3d(total_channels, total_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, M, C, D, H, W)
        x = rearrange(x, 'b m c d h w -> b (m c) d h w')
        return x * self.gate(x)


class NestedFormer(nn.Module):
    def __init__(self, model_num,
                 out_channels,
                 image_size,
                 fea=(16, 16, 32, 64, 128, 16),
                 window_size=(2, 4, 4),
                 pool_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 self_num_layer=2,
                 token_mixer_size=32,
                 token_learner=True):

        super().__init__()
        self.out_channels = out_channels
        self.model_num = model_num
        self.pool_size = pool_size

        pool_size_all = [1, 1, 1]
        image_size_s = [image_size]
        for p in pool_size:
            pool_size_all = [pool_size_all[i] * p[i] for i in range(len(p))]
            image_size_s.append((image_size_s[-1][0] // p[0],
                                 image_size_s[-1][1] // p[1],
                                 image_size_s[-1][2] // p[2]))
        new_image_size = [image_size[i] // pool_size_all[i] for i in range(3)]

        self.encoder = Encoder(model_num=model_num,
                               img_size=image_size_s[1:],
                               fea=fea,
                               pool_size=pool_size)

        self.fusion = NMaFaLayer(model_num=model_num,
                                 in_channels=fea[4],
                                 hidden_size=fea[4],
                                 img_size=new_image_size,
                                 mlp_size=2 * fea[4],
                                 self_num_layer=self_num_layer,
                                 window_size=window_size,
                                 token_mixer_size=token_mixer_size,
                                 token_learner=token_learner)

        self.fusion_conv_5 = TwoConv(fea[4], fea[4], 3, 1, 1)
        self.fusion_conv_1 = TwoConv(model_num * fea[0], fea[0], 3, 1, 1)
        self.fusion_conv_2 = TwoConv(model_num * fea[1], fea[1], 3, 1, 1)
        self.fusion_conv_3 = TwoConv(model_num * fea[2], fea[2], 3, 1, 1)
        self.fusion_conv_4 = TwoConv(model_num * fea[3], fea[3], 3, 1, 1)

        # ✅ MSG blocks
        self.msg_2 = ModalityGate(fea[1], model_num=self.model_num)
        self.msg_3 = ModalityGate(fea[2], model_num=self.model_num)
        self.msg_4 = ModalityGate(fea[3], model_num=self.model_num)

        self.upcat_4 = UpCat(fea[4], fea[3], fea[3], pool_size=pool_size[3])
        self.upcat_3 = UpCat(fea[3], fea[2], fea[2], pool_size=pool_size[2])
        self.upcat_2 = UpCat(fea[2], fea[1], fea[1], pool_size=pool_size[1])
        self.upcat_1 = UpCat(fea[1], fea[0], fea[5], pool_size=pool_size[0])

        # ✅ Final conv for deep supervision
        self.final_conv = nn.Conv3d(fea[5], out_channels, 1, 1)
        self.deep_conv_2 = nn.Conv3d(fea[1], out_channels, 1, 1)
        self.deep_conv_3 = nn.Conv3d(fea[2], out_channels, 1, 1)
        self.deep_conv_4 = nn.Conv3d(fea[3], out_channels, 1, 1)

    def forward(self, x):
        assert x.shape[1] == self.model_num

        encoder_x = self.encoder(x)

        encoder_1 = torch.stack([encoder_x[i][4] for i in range(self.model_num)], dim=1)
        encoder_2 = torch.stack([encoder_x[i][3] for i in range(self.model_num)], dim=1)
        encoder_3 = torch.stack([encoder_x[i][2] for i in range(self.model_num)], dim=1)
        encoder_4 = torch.stack([encoder_x[i][1] for i in range(self.model_num)], dim=1)
        encoder_5 = torch.stack([encoder_x[i][0] for i in range(self.model_num)], dim=1)

        # High-level fusion
        fusion_out = self.fusion(encoder_5)
        encoder_5 = encoder_5.mean(dim=1)
        fusion_out_cnn = self.fusion_conv_5(encoder_5)
        fusion_out = fusion_out.view(fusion_out_cnn.shape)
        fusion_out = fusion_out + fusion_out_cnn

        # MSG-based low-level fusion
        encoder_2 = self.msg_2(encoder_2)
        encoder_3 = self.msg_3(encoder_3)
        encoder_4 = self.msg_4(encoder_4)

        encoder_1 = rearrange(encoder_1, "b n c d w h -> b (n c) d w h")

        encoder_2_cnn = self.fusion_conv_2(encoder_2)
        encoder_3_cnn = self.fusion_conv_3(encoder_3)
        encoder_4_cnn = self.fusion_conv_4(encoder_4)
        encoder_1_cnn = self.fusion_conv_1(encoder_1)

        u4 = self.upcat_4(fusion_out, encoder_4_cnn)
        u3 = self.upcat_3(u4, encoder_3_cnn)
        u2 = self.upcat_2(u3, encoder_2_cnn)
        u1 = self.upcat_1(u2, encoder_1_cnn)

        # ✅ Deep supervision outputs
        out_main = self.final_conv(u1)
        out_deep2 = self.deep_conv_2(u2)
        out_deep3 = self.deep_conv_3(u3)
        out_deep4 = self.deep_conv_4(u4)

        return [out_main, out_deep2, out_deep3, out_deep4]
