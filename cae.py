import torch
import torch.nn as nn

class CAE(nn.Module): # convolutional autoencoder
    def __init__(self, in_channel, out_channel=32, kernel_size=3, stride=1, padding=0):
        super(CAE,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.encoder = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(out_channel, in_channel, kernel_size, stride, padding))
        self.pooling = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)
        self.unpooling = nn.MaxUnpool2d(kernel_size, stride, padding)

    def forward(self,x):
        x = self.encoder(x)
        x, indices = self.pooling(x)
        x = self.unpooling(x, indices) # filter out part of information since non-maximal values are lost
        x = self.decoder(x)
        return x

    def extract_feature_raw(self, x_data, batch_size, window_length, device):
        batched_x_data = []
        batched_x_feature = []
        for idx in range(0, len(x_data)-batch_size, batch_size):
            sample = torch.tensor(x_data[idx:idx+batch_size, :, :]).reshape(batch_size, 1, window_length, -1).float()
            batched_x_data.append(sample)
            sample = sample.to(device)
            sample_embedding = self.encoder(sample)
            batched_x_feature.append(sample_embedding)
        return batched_x_feature, batched_x_data

    def extract_feature_embed(self, data):
        result = []
        for idx in range(len(data)):
            sample_embedding = self.encoder(data[idx])
            result.append(sample_embedding)
        return result

class UNet_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    def forward():
        return self.relu(self.conv2(self.relu(self.conv1(x))))
    
class UNet_Encoder(nn.Module):
    def __init__(self, chs=(1, 8, 16, 32)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([UNet_Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)
    def forward():
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class UNet_Decoder(nn.Module):
    def __init__(self, chs = (32, 16, 8), cond_embed = None):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([UNet_Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.crossAttention = ExternalCrossAttention(d_model = 32, S = 128)
    def forward(self, x, encoder_features, embed_condition):
        x = self.crossAttention(x, embed_condition)
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim = 1) # note here is cat., so channel doubled
            x = self.dec_blocks[i](x)
        return x
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs
                                      
class UNet(nn.Module):
    def __init__(self, enc_chs=(1, 8, 16, 32), dec_chs=(32, 16, 8), num_class = 5, retrain_dim = False, out_sz = (1, 100)):
        super().__init__()
        self.encoder = UNet_Encoder(enc_chs)
        self.decoder = UNet_Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
    
    def forward(self, x, embed_condition):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:], embed_condition)
        # out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        return out

class ExternalCrossAttention(nn.Module):
    def __init__(self, d_model,S=64):
        super().__init__()
        self.mq = nn.Linear(d_model, S, bias=False)
        self.mk = nn.Linear(d_model,S,bias=False)
        self.mv = nn.Linear(S,d_model,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, latent_unet, latent_condition):
        Q = self.mq(latent_unet)
        K = self.mk(latent_condition)
        V = self.mv(latent_condition)
        attn=torch.bmm(Q, K)
        attn=self.softmax(attn)
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=torch.bmm(attn, V)
        return out