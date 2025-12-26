import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA


s0 = torch.Tensor([94.80, 104.80, 105.90, 96.80, 113.90, 125.60, 125.50, 121.30, 121.30,
                   113.50, 113.10, 110.80, 106.50, 108.80, 105.30, 104.40, 100.00, 96.00, 95.10, 89.10,
                   90.50, 90.30, 88.40, 84.00, 85.10, 81.90, 82.60, 84.90, 81.30, 71.90, 74.30, 76.40,
                   63.30])

s1 = torch.Tensor([43.40, 46.30, 43.90, 37.10, 36.70, 35.90, 32.60, 27.90, 24.30, 20.10,
                   16.20, 13.20, 8.60, 6.10, 4.20, 1.90, 0.00, -
                   1.60, -3.50, -3.50, -5.80, -7.20, -8.60,
                   -9.50, -10.90, -10.70, -12.00, -14.00, -13.60, -12.00, -13.30, -12.90, -10.60
                   ])
s2 = torch.Tensor([-1.1, -0.5, -0.7, -1.2, -2.6, -2.9, -2.8, -2.6, -2.6, -1.8, -1.5, -1.3,
                   -1.2, -1.0, -0.5, -0.3, 0.0, 0.2, 0.5, 2.1, 3.2, 4.1, 4.7, 5.1, 6.7, 7.3, 8.6, 9.8, 10.2,
                   8.3, 9.6, 8.5, 7.0])


class Decoder(torch.nn.Module):
    def __init__(self, device, n_components=2, lightVectorSize=15, size=64, util_dir="./"):
        super(Decoder, self).__init__()
        self.lightVectorSize = lightVectorSize
        self.size = size

        illA = sio.loadmat(f"{util_dir}util/illumA.mat")
        illA = illA['illumA'][0][0]

        illA = illA / illA.sum()
        self.illA = torch.tensor(illA).to(device)

        self.s0 = s0.to(device)
        self.s1 = s1.to(device)
        self.s2 = s2.to(device)

        self.illF = torch.Tensor(sio.loadmat(
            f'{util_dir}util/illF')['illF']).to(device)[0]
        self.illF = self.illF / torch.reshape(torch.sum(self.illF, 0), (1, 12))

        rgbData = sio.loadmat(f"{util_dir}util/rgbCMF.mat")
        cameraSensitivityData = np.array(list(np.array(rgbData['rgbCMF'][0])))
        pca = PCA(n_components)

        Y = np.transpose(cameraSensitivityData, (2, 0, 1))

        for camera in range(28):
            for channel in range(3):
                # should use max but doesn't matter since white balance divides
                Y[camera, channel] /= np.sum(Y[camera, channel])

        Y = np.resize(Y, (28, 99))

        pca.fit(Y)

        pcaComponents = pca.components_ * \
            np.resize(pca.explained_variance_ ** 0.5, (n_components, 1))
        # Done so that vector is on the same scale as matlab
        pcaComponents[1] *= -1

        self.pcaMeans = torch.reshape(torch.tensor(
            pca.mean_), (1, 99)).float().to(device)
        self.pcaComponents = torch.tensor(
            pcaComponents).permute(1, 0).float().to(device)

        Newskincolour = sio.loadmat(f'{util_dir}util/Newskincolour.mat')['Newskincolour']
        Newskincolour = Newskincolour.transpose((2, 0, 1))
        skinColor = torch.tensor(Newskincolour).to(device)
        self.skinColor = torch.reshape(skinColor, (1, 33, 256, 256))

        tmatrix = sio.loadmat(f"{util_dir}util/Tmatrix.mat")['Tmatrix']
        tmatrix = np.transpose(tmatrix, (2, 0, 1))
        tmatrix = torch.tensor(tmatrix).to(device)
        self.tmatrix = torch.reshape(tmatrix, (1, 9, 128, 128))
        self.txyx2rgb = torch.tensor([[3.2406, -1.537, -0.498],
                                      [-0.968, 1.8758, 0.0415],
                                      [0.0557, -0.204, 1.0570]]
                                     ).to(device)

    def chromacity(self, t):
        t = t * 21000
        t = t + 4000

        x1 = -4.6070 * (10 ** 9) / (t ** 3) + (2.9678 * 10 ** 6) / \
            (t ** 2) + (0.09911 * 10 ** 3) / t + 0.244063
        x2 = -2.0064 * (10 ** 9) / (t ** 3) + (1.9018 * 10 ** 6) / \
            (t ** 2) + (0.24748 * 10 ** 3) / t + 0.237040

        x = (t <= 7000) * x1 + (t > 7000) * x2

        y = -3 * x ** 2 + 2.87 * x - 0.275

        return x, y

    def illuminanceD(self, temp):
        x, y = self.chromacity(temp)

        m = 0.0241 + 0.2562 * x - 0.7341 * y
        m1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / m
        m2 = (0.0300 - 31.4424 * x + 30.0717 * y) / m

        s = self.s0 + m1 * self.s1 + m2 * self.s2
        return s / torch.reshape(torch.sum(s, 1), (-1, 1))

    def __call__(self, lighting, features):
        lighting_parameters = lighting[:, :self.lightVectorSize]
        b = lighting[:, self.lightVectorSize:]

        mel, blood, shade, spec = features

        ########### Scaling ###########

        lighting_weights = lighting_parameters[:, :14]  # 前14个是权重
        lighting_weights = F.softmax(lighting_weights, 1) # 前14个经过softmax
        weightA = lighting_weights[:, 0]               # 白炽灯权重
        weightA = torch.reshape(weightA, (-1, 1))
        weightD = lighting_weights[:, 1]               # 日光灯权重
        weightD = torch.reshape(weightD, (-1, 1))
        fWeights = lighting_weights[:, 2:14]           # 荧光灯权重
        colorTemp = torch.sigmoid(lighting_parameters[:, 14]) # 色温，经过sigmoid函数
        colorTemp = torch.reshape(colorTemp, (-1, 1))

        b = 6 * torch.sigmoid(b) - 3  # 相机统计模型的参数，经过sigmoid函数

        # mel = -2 * torch.sigmoid(mel) + 1          # -1~1
        # blood = -2 * torch.sigmoid(blood) + 1      # -1~1
        mel = 2 * torch.sigmoid(mel) - 1          # -1~1
        blood = 2 * torch.sigmoid(blood) - 1      # -1~1
        shade = torch.exp(shade)                   # 0~inf
        spec = torch.exp(spec)                     # 0~inf

        ########### Illumination ###########
        '''
        Inputs:
            weightA, weightD                :   [N]
            Fweights                        :   [N,12]
            CCT                             :   [N]
            illumA                          :   [33]
            illumDNorm                      :   [33,22]
            illumFNorm                      :   [33,12]
        Output:
            e                               :   [N,33]
        '''
        # illumination A: [N,33] = [N,1]*[1,33]
        aLightVector = weightA * self.illA
        # illumination D: [N,33]
        dLightVector = weightD * self.illuminanceD(colorTemp)
        # illumination F:
        fLightVector = F.linear(fWeights, self.illF)

        e = aLightVector + dLightVector + fLightVector  # 公式11
        # print(f"e.shape: {e.shape}, aLightVector: {aLightVector.shape}, dLightVector: {dLightVector.shape}, fLightVector: {fLightVector.shape}")
        eSums = torch.reshape(torch.sum(e, 1), (-1, 1))

        e = e / eSums        # 归一化


        S = F.linear(b, self.pcaComponents)   # 相机统计模型公式10，乘以主成分
        S += self.pcaMeans   # 相机统计模型公式10，加上均值

        S = F.relu(S)   # (BS, 99)

        S = torch.reshape(S, (-1, 3, 33)) # (BS, 3, 33)

        lightColor = S * torch.reshape(e, (-1, 1, 33)) # 相机统计模型乘以光照模型
        lightColor = torch.sum(S, 2)   # (BS, 3)

        ########### Specularities ###########

        spec = spec * torch.reshape(lightColor, (-1, 3, 1, 1))

        ########### Diffuse ###########

        bioPhysicalLayer = torch.cat((mel, blood), 1).permute((0, 2, 3, 1)) # (BS, 64, 64, 2)
        skinColorGrid = self.skinColor.repeat(
            (bioPhysicalLayer.shape[0], 1, 1, 1))  # (BS, 33, 64, 64)
        r_total = F.grid_sample(skinColorGrid, bioPhysicalLayer, align_corners=True) # input = (BS, 33, 64, 64) grid =  (BS, 64, 64, 2)
        # r_total = (BS, 33, 64, 64)
        spectra = r_total * torch.reshape(e, (-1, 33, 1, 1))  # 面部颜色与光照模型相乘
        spectra = torch.reshape(spectra, (-1, 1, 33, self.size, self.size)) # 添加了一个维度

        S = torch.reshape(S, (-1, 3, 33, 1, 1)) # 相机统计模型

        diffuse = torch.sum(spectra * S, 2)  # 面部颜色、光照模型、相机统计模型相乘
        diffuse = shade * diffuse  # 添加阴影 (BS, 1, 64, 64) * (BS, 3, 64, 64)

        raw = diffuse + spec   # 添加反射光谱

        ########### Camera Transformation ###########

        wb = raw / torch.reshape(lightColor, (-1, 3, 1, 1)) # (BS, 3, 64, 64) / (BS, 3, 1, 1) 白平衡化

        tMatrixGrid = self.tmatrix.repeat((wb.shape[0], 1, 1, 1)) # (BS, 9, 128, 128)
        bIndex = torch.reshape(b / 3, (-1, 1, 1, 2))  # (BS, 1, 1, 2)

        ts = F.grid_sample(tMatrixGrid, bIndex, align_corners=True)  # (BS, 9, 1, 1)
        ts = torch.reshape(ts, (-1, 9, 1, 1)) # (BS, 9, 1, 1)

        # 转到XYZ空间，ts的0，3，6是X，1，4，7是Y，2，5，8是Z
        ix = ts[:, 0, :, :] * wb[:, 0, :, :] + ts[:, 3, :, :] * \
            wb[:, 1, :, :] + ts[:, 6, :, :] * wb[:, 2, :, :] # (BS, 64, 64)
        iy = ts[:, 1, :, :] * wb[:, 0, :, :] + ts[:, 4, :, :] * \
            wb[:, 1, :, :] + ts[:, 7, :, :] * wb[:, 2, :, :]
        iz = ts[:, 2, :, :] * wb[:, 0, :, :] + ts[:, 5, :, :] * \
            wb[:, 1, :, :] + ts[:, 8, :, :] * wb[:, 2, :, :]

        ix = torch.reshape(ix, (-1, 1, self.size, self.size))
        iy = torch.reshape(iy, (-1, 1, self.size, self.size))
        iz = torch.reshape(iz, (-1, 1, self.size, self.size))

        xyz = torch.cat((ix, iy, iz), 1)  # xyz图像

        xyz = xyz.permute((0, 2, 3, 1))
        rgb = F.linear(xyz, self.txyx2rgb)  # xyz2rgb
        rgb = rgb.permute((0, 3, 1, 2))

        rgb = F.relu(rgb) # 重建图像

        shade = torch.reshape(shade, (-1, self.size, self.size))
        blood = torch.reshape(blood, (-1, self.size, self.size))
        mel = torch.reshape(mel, (-1, self.size, self.size))
        # print(rgb.shape, shade.shape, spec.shape, blood.shape, mel.shape, b.shape)
        return rgb, shade, spec, blood, mel, b, raw


class PhysNet_padding_ED_peak(nn.Module):
    def __init__(self, in_channels=3, frames=160):
        super(PhysNet_padding_ED_peak, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1], stride=[2, 1, 1],
                               padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 2, [1, 1, 1], stride=1, padding=0) # NOTE : temp 2

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # x [3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]
        # print(x.shape)
        x = self.poolspa(x)  # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG_peak = x.squeeze(-1).squeeze(-1)  # [Batch, 2, T]

        rPPG_peak = rPPG_peak[:, 0, :]
        rPPG_peak = (rPPG_peak - torch.mean(rPPG_peak)) / torch.abs(rPPG_peak).max()  # normalize

        return rPPG_peak, x_visual, x_visual3232, x_visual1616


channels = [3, 32, 64, 128]


class Unet(nn.Module):
    def __init__(self, n_components=2, lightVectorSize=15, size=64):
        # n_components: 光照SPD的维度
        super(Unet, self).__init__()


        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        # )
        self.convolutions = nn.ModuleList()
        self.encoderBatchnorms = nn.ModuleList()
        for i in range(1, len(channels)):
            self.convolutions.append(
                nn.Conv2d(channels[i - 1], channels[i], 3, padding=1))
            self.encoderBatchnorms.append(nn.BatchNorm2d(channels[i]))

            self.convolutions.append(
                nn.Conv2d(channels[i], channels[i], 3, padding=1))
            self.encoderBatchnorms.append(nn.BatchNorm2d(channels[i]))

            self.convolutions.append(
                nn.Conv2d(channels[i], channels[i], 3, padding=1))
            self.encoderBatchnorms.append(nn.BatchNorm2d(channels[i]))

            if i != len(channels) - 1:
                size //= 2

        self.low_resolution = size

        self.fc1 = nn.Linear(channels[-1] * size * size, channels[-1])
        self.batchnorm1 = nn.BatchNorm1d(channels[-1])
        self.fc2 = nn.Linear(channels[-1], channels[-1])
        self.batchnorm2 = nn.BatchNorm1d(channels[-1])
        self.fc3 = nn.Linear(channels[-1], lightVectorSize + n_components) # 最后一层全连接层，（相机光谱灵敏度、光照SPD

        self.decoderConvolutions = nn.ModuleList()
        self.decoderBatchnorms = nn.ModuleList()

        # for _ in range(4):
        convs = nn.ModuleList()
        bns = nn.ModuleList()
        for i in reversed(range(1, len(channels) - 1)):
            size *= 2

            convs.append(
                nn.Conv2d(channels[i] + channels[i + 1], channels[i], 3, padding=1))
            bns.append(nn.BatchNorm2d(channels[i]))
            convs.append(nn.Conv2d(channels[i], channels[i], 3, padding=1))
            bns.append(nn.BatchNorm2d(channels[i]))
            convs.append(nn.Conv2d(channels[i], channels[i], 3, padding=1))
            bns.append(nn.BatchNorm2d(channels[i]))

        convs.append(nn.Conv2d(channels[1], 4, 3, padding=1))

        self.decoderConvolutions = convs
        self.decoderBatchnorms = bns

    def forward(self, x):
        image = x  # (B, 3, 64, 64)

        skipValues = []
        # image = self.conv(image)
        # image = F.max_pool2d(image, 2)
        ########### Encoding ###########
        for convIndex in range(len(self.convolutions)):
            image = self.convolutions[convIndex](image)
            image = self.encoderBatchnorms[convIndex](image)
            image = F.relu(image)

            if convIndex % 3 == 2 and convIndex != len(self.convolutions) - 1:
                skipValues.append(torch.clone(image))
                image = F.max_pool2d(image, 2)

        skipValues.reverse()

        ########### Fully Connected Layer ###########
        # (B, channels[-1], low_resolution, low_resolution) -> (B, channels[-1], * low_resolution *low_resolution)
        lighting = torch.reshape(
            image, (-1, self.low_resolution * self.low_resolution * channels[-1]))
        lighting = self.fc1(lighting)
        lighting = self.batchnorm1(lighting)
        lighting = F.relu(lighting)
        lighting = self.fc2(lighting)
        lighting = self.batchnorm2(lighting)
        lighting = F.relu(lighting)
        lighting = self.fc3(lighting)

        features = []

        ########### Decoding ###########

        # for out in range(4):
        feature = torch.clone(image) # 每一个decoder分支都是一个clone

        for i in range(len(self.decoderConvolutions) - 1):
            if i % 3 == 0:
                feature = F.interpolate(feature, scale_factor=2)
                feature = torch.cat((feature, skipValues[i // 3]), 1)

            feature = self.decoderConvolutions[i](feature)
            feature = self.decoderBatchnorms[i](feature)
            feature = F.relu(feature)
        # feature = F.interpolate(feature, scale_factor=2)

        feature = self.decoderConvolutions[-1](feature)

        features = feature.chunk(4, 1)
        return lighting, features

class Shading(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # input: [B, 3, T, H, W]
        # output: [B, 1, H, W]
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 16, 16))

        # 转置卷积
        self.ConvBlock7 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )


        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)


    def forward(self, x):
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [32, T/2, 32,32]    Temporal halve

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]

        x = self.avgpool3d(x)  # x [64, 1, 16,16]

        x = x.squeeze(-3)  # x [64, 16,16]
        x = self.ConvBlock7(x)  # x [32, 16,16]
        x = self.ConvBlock8(x)  # x [16, 16,16]
        x = self.ConvBlock9(x)  # x [1, 16,16]
        return x # [B, 1, 64, 64]

class CrossAttention(nn.Module):
    def __init__(self, dim=16, num_heads=4, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x, blood):
        # B C T H W

        # x = x.flatten(2).transpose(1, 2)
        # blood = blood.flatten(2).transpose(1, 2) # B N C
        B, N, C = x.shape
        kv = self.kv1(x)
        # print("kv", kv.shape)
        xk, xv = kv.reshape(kv.shape[0], kv.shape[1], 2, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # print("kv", xk.shape, xv.shape)
        blood_kv = self.kv2(blood)
        blood_k, blood_v = blood_kv.reshape(blood_kv.shape[0], blood_kv.shape[1], 2, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (xk.transpose(-2, -1) @ xv) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        # ctx1 = F.relu(ctx1)
        ctx2 = (blood_k.transpose(-2, -1) @ blood_v) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        # ctx2 = F.relu(ctx2)
        x = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        blood = blood.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        x = (x @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        blood = (blood @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()


        return x, blood




class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.norm2 = nn.BatchNorm3d(dim)

    def forward(self, x1, x2):
        B, C, T, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)

        v1, v2 = self.cross_attn(x1, x2)

        x1 = x1.transpose(1, 2).reshape(B, C, T, H, W)
        x2 = x2.transpose(1, 2).reshape(B, C, T, H, W)
        v1 = v1.transpose(1, 2).reshape(B, C, T, H, W)
        v2 = v2.transpose(1, 2).reshape(B, C, T, H, W)
        v1 = self.norm1(v1)
        v2 = self.norm2(v2)
        out_x1 = x1 + v1
        out_x2 = x2 + v2

        return out_x1, out_x2


class MultiModalNet(torch.nn.Module):
    def __init__(self, frames=160):
        super().__init__()
        self.frames = frames

        self.video_encoder = PhysNet_padding_ED_peak(in_channels=4, frames=frames)
        self.blood_encoder = PhysNet_padding_ED_peak(in_channels=1, frames=frames)
        self.cross_1 = CrossPath(16, num_heads=1)
        self.cross_2 = CrossPath(64, num_heads=4)
        self.cross_3 = CrossPath(64, num_heads=4)
        self.cross_4 = CrossPath(64, num_heads=4)


    def forward(self, x, blood):

        x = self.video_encoder.ConvBlock1(x)
        x = self.video_encoder.MaxpoolSpa(x)
        blood = self.blood_encoder.ConvBlock1(blood)
        blood = self.blood_encoder.MaxpoolSpa(blood)  # 160, 32, 32
        video_before_1, blood_before_1 = x, blood
        x, blood = self.cross_1(x, blood)
        video_after_1, blood_after_1 = x, blood

        x = self.video_encoder.ConvBlock2(x)
        x = self.video_encoder.ConvBlock3(x)
        x = self.video_encoder.MaxpoolSpaTem(x)
        blood = self.blood_encoder.ConvBlock2(blood)
        blood = self.blood_encoder.ConvBlock3(blood)
        blood = self.blood_encoder.MaxpoolSpaTem(blood)  # 80, 32, 32
        video_before_2, blood_before_2 = x, blood
        x, blood = self.cross_2(x, blood)
        video_after_2, blood_after_2 = x, blood

        x = self.video_encoder.ConvBlock4(x)
        x = self.video_encoder.ConvBlock5(x)
        x = self.video_encoder.MaxpoolSpaTem(x)
        blood = self.blood_encoder.ConvBlock4(blood)
        blood = self.blood_encoder.ConvBlock5(blood)
        blood = self.blood_encoder.MaxpoolSpaTem(blood)  # 40, 16, 16
        x, blood = self.cross_3(x, blood)


        x = self.video_encoder.ConvBlock6(x)
        x = self.video_encoder.ConvBlock7(x)
        blood = self.blood_encoder.ConvBlock6(blood)
        blood = self.blood_encoder.ConvBlock7(blood)
        x, blood = self.cross_4(x, blood)



        # out
        x = self.video_encoder.upsample(x)
        x = self.video_encoder.upsample2(x)
        x = self.video_encoder.poolspa(x)
        x = self.video_encoder.ConvBlock10(x)
        x = x.squeeze(-1).squeeze(-1)  # [Batch, 2, T]
        x = x[:, 0, :]

        blood = self.blood_encoder.upsample(blood)
        blood = self.blood_encoder.upsample2(blood)
        blood = self.blood_encoder.poolspa(blood)
        blood = self.blood_encoder.ConvBlock10(blood)
        blood = blood.squeeze(-1).squeeze(-1)  # [Batch, 2, T]
        blood = blood[:, 0, :]

        rppg = x + blood
        rppg = (rppg - torch.mean(rppg)) / torch.abs(rppg).max()
        
        visual = {
            'video_before_1' : video_before_1,
            'video_after_1' : video_after_1,
            'video_before_2' : video_before_2,
            'video_after_2' : video_after_2,
            'blood_before_1' : blood_before_1,
            'blood_after_1' : blood_after_1,
            'blood_before_2' : blood_before_2,
            'blood_after_2' : blood_after_2,
        }

        return rppg, x, blood, visual

class BioPhysNet(torch.nn.Module):
    def __init__(self, frames, encoder_dict="/root/opensource/rppg/using_file/bioface/best_model.pth", device="cpu") -> None:
        super().__init__()
        self.n_components = 2
        self.lightVectorSize = 15
        self.encoder = Unet(self.n_components, self.lightVectorSize)
        self.decoder = Decoder(device, self.n_components, self.lightVectorSize, util_dir="./using_file/bioface/")
        self.rppg_model = MultiModalNet(frames=frames)
        self.shading = Shading()

    def forward(self, batch):
        video = batch["video"]
        video_length = video.shape[2]
        image = video.transpose(1, 2) # [B, C, T, H, W] -> [B, T, C, H, W]
        image = image.reshape(-1, image.shape[2], image.shape[3], image.shape[4]) # [B, T, C, H, W] -> [B*T, C, H, W]
        lighting, features = self.encoder(image)
        rgb, shade, spec, blood, mel, b, diffuse  = self.decoder(lighting, features)
    
        # 将血红蛋白和视频拼接起来
        blood_video = blood.reshape(-1, video_length, 1, blood.shape[1], blood.shape[2]) # [B*T, H, W] -> [B, T, 1, H, W]
        blood_video = blood_video.transpose(1, 2) # [B, T, C, H, W] -> [B, C, T, H, W]


        cat_video = torch.cat((video, blood_video), dim=1)
        pred, _, _, visual = self.rppg_model(cat_video, blood_video)
        return {"rppg": pred, "rgb": rgb, "shade": shade, "spec": spec, "blood": blood, "mel": mel, "b": b, "image": image, "visual" : visual}


def load_model(args, device):
    return BioPhysNet(frames=args.T, device=device)

