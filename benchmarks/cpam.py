import torch
import torch.nn as nn
import torch.nn.functional as F

class CPAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(CPAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.l = nn.Parameter(torch.zeros((1,196)))
        self.G = nn.Sequential(
                nn.Linear(in_features=150*1024, out_features=1024),
                )
        #self.R =i
    def forward(self, x, text_embed):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        #x.shape : torch.Size([8, 1024, 14, 14]), text_embed.shape: torch.Size([8, 1, 150, 1024])
        text_embed = torch.squeeze(text_embed, 1) #text_embed.shape: torch.Size([8, 150, 1024])
        text_embed = torch.flatten(text_embed, start_dim=1) #text_embed.shape: torch.Size([8, 150*1024])
        text_embed = self.G(text_embed) #text_embed.shape: torch.Size([8, 1024])
        text_embed = torch.unsqueeze(text_embed, 2) #text_embed.shape: torch.Size([8, 1024, 1])
        
        l_tmp = self.l.repeat(text_embed.shape[0], 1, 1)
        text_embed = torch.bmm(text_embed, l_tmp)#text_embed.shape: torch.Size([8, 1, 196])
        #print(text_embed.shape)
        text_embed = torch.reshape(text_embed, x.shape) #text_embed.shape: torch.Size([8, 1024, 14, 14])

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(text_embed).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(text_embed).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class UnetCPAM(torch.nn.Module):
    def __init__(self, config):
        super(UnetCPAM, self).__init__()

        self.n_channels = config["in_channels"]
        self.n_classes = config["out_channels"]
        self.bilinear = config["bilinear"]
        factor = 2 if self.bilinear else 1

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, self.bilinear)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up2 = Up(512, self.bilinear)
        self.up_conv2 = DoubleConv(512, 256)

        self.up3 = Up(256, self.bilinear)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = Up(128, self.bilinear)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, self.n_classes)
        self.sigmoid = nn.Sigmoid()
        self.cpam = CPAM_Module(1024)

    def forward(self, img, text_embed):

        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        cpam_out = self.cpam(x5, text_embed) 

        decode1 = self.up1(cpam_out)

        x = concatenate_layers(decode1, x4)
        x = self.up_conv1(x)

        decode2 = self.up2(x)

        x = concatenate_layers(decode2, x3)
        x = self.up_conv2(x)

        decode3 = self.up3(x)

        x = concatenate_layers(decode3, x2)
        x = self.up_conv3(x)

        decode4 = self.up4(x)

        x = concatenate_layers(decode4, x1)
        x = self.up_conv4(x)

        logits = self.outc(x)

        return logits


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)

        return x1


def concatenate_layers(x1, x2):
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
