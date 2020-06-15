from __future__ import print_function, division
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import pretrainedmodels
import numpy as np
import numpy.matlib
import pickle

# import gluoncvth as gcv
from lib.torch_util import Softmax1D
from lib.conv4d import Conv4d


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = (
        torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5)
        .unsqueeze(1)
        .expand_as(feature)
    )
    return torch.div(feature, norm)


class FeatureExtraction(torch.nn.Module):
    def get_feature_backbone(self, model):
        resnet_feature_layers = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]
        last_layer = "layer3"
        resnet_module_list = [getattr(model, l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index(last_layer)
        model = nn.Sequential(*resnet_module_list[: last_layer_idx + 1])
        return model

    def __init__(
        self,
        train_fe=False,
        feature_extraction_cnn="resnet101",
        feature_extraction_model_file="",
        normalization=True,
        last_layer="",
        use_cuda=True,
    ):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        self.feature_extraction_cnn = feature_extraction_cnn
        if feature_extraction_cnn == "vgg":
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to indicated layer
            vgg_feature_layers = [
                "conv1_1",
                "relu1_1",
                "conv1_2",
                "relu1_2",
                "pool1",
                "conv2_1",
                "relu2_1",
                "conv2_2",
                "relu2_2",
                "pool2",
                "conv3_1",
                "relu3_1",
                "conv3_2",
                "relu3_2",
                "conv3_3",
                "relu3_3",
                "pool3",
                "conv4_1",
                "relu4_1",
                "conv4_2",
                "relu4_2",
                "conv4_3",
                "relu4_3",
                "pool4",
                "conv5_1",
                "relu5_1",
                "conv5_2",
                "relu5_2",
                "conv5_3",
                "relu5_3",
                "pool5",
            ]
            if last_layer == "":
                last_layer = "pool4"
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(
                *list(self.model.features.children())[: last_layer_idx + 1]
            )
        # for resnet below
        resnet_feature_layers = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]
        if feature_extraction_cnn == "resnet101":
            self.model = models.resnet101(pretrained=True)
            self.model = self.get_feature_backbone(self.model)

        if feature_extraction_cnn == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.model = self.get_feature_backbone(self.model)

        if feature_extraction_cnn == "resnet152":
            model_name = "resnet152"  # could be fbresnet152 or inceptionresnetv2
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained="imagenet"
            )
            self.model = self.get_feature_backbone(model)

        if feature_extraction_cnn == "resnet101fcn":
            self.model = gcv.models.get_fcn_resnet101_voc(pretrained=True)

        if feature_extraction_cnn == "resnext101":
            model_name = "resnext101_32x4d"  # could be fbresnet152 or inceptionresnetv2
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained="imagenet"
            )
            self.model = model.features[:-1]

        if feature_extraction_cnn == "resnext101_64x4d":
            model_name = "resnext101_64x4d"
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained="imagenet"
            )
            self.model = model.features[:-1]

        if feature_extraction_cnn == "resnet101fpn":
            if feature_extraction_model_file != "":
                resnet = models.resnet101(pretrained=True)
                # swap stride (2,2) and (1,1) in first layers (PyTorch ResNet is slightly different to caffe2 ResNet)
                # this is required for compatibility with caffe2 models
                resnet.layer2[0].conv1.stride = (2, 2)
                resnet.layer2[0].conv2.stride = (1, 1)
                resnet.layer3[0].conv1.stride = (2, 2)
                resnet.layer3[0].conv2.stride = (1, 1)
                resnet.layer4[0].conv1.stride = (2, 2)
                resnet.layer4[0].conv2.stride = (1, 1)
            else:
                resnet = models.resnet101(pretrained=True)
            resnet_module_list = [getattr(resnet, l) for l in resnet_feature_layers]
            conv_body = nn.Sequential(*resnet_module_list)
            self.model = fpn_body(
                conv_body,
                resnet_feature_layers,
                fpn_layers=["layer1", "layer2", "layer3"],
                normalize=normalization,
                hypercols=True,
            )
            if feature_extraction_model_file != "":
                self.model.load_pretrained_weights(feature_extraction_model_file)

        if feature_extraction_cnn == "densenet201":
            self.model = models.densenet201(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children())[:-3])

        if train_fe == False:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, image_batch):
        if self.feature_extraction_cnn == "resnet101fcn":
            features = self.model(image_batch)
            features = torch.cat((features[0], features[1]), 1)
        else:
            features = self.model(image_batch)

        if self.normalization and not self.feature_extraction_cnn == "resnet101fpn":
            features = featureL2Norm(features)

        return features


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        b, c, hA, wA = feature_A.size()
        b, c, hB, wB = feature_B.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # size [b,c,h*w]
        feature_B = feature_B.view(b, c, hB * wB)  # size [b,c,h*w]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_A, feature_B)
        # indexed [batch,row_A,col_A,row_B,col_B]
        correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class SpatialContextNet(torch.nn.Module):
    def __init__(self, kernel_size=5, output_channel=1024, use_cuda=True):
        super(SpatialContextNet, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = torch.nn.Conv2d(
            1024 + self.kernel_size * self.kernel_size,
            output_channel,
            1,
            bias=True,
            padding_mode="zeros",
        )
        if use_cuda:
            self.conv = self.conv.cuda()

    def forward(self, feature):
        b, c, h, w = feature.size()
        feature_normalized = F.normalize(feature, p=2, dim=1)
        feature_pad = F.pad(
            feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0
        )
        output = torch.zeros(
            [self.kernel_size * self.kernel_size, b, h, w],
            dtype=feature.dtype,
            requires_grad=feature.requires_grad,
        )
        if feature.is_cuda:
            output = output.cuda(feature.get_device())
        for c in range(self.kernel_size):
            for r in range(self.kernel_size):
                output[c * self.kernel_size + r] = (
                    feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized
                ).sum(1)

        output = output.transpose(0, 1).contiguous()
        output = torch.cat((feature, output), 1)
        output = self.conv(output)
        output = F.relu(output)
        return output


class Pairwise(torch.nn.Module):
    def __init__(self, context_size=5, output_channel=128, use_cuda=True):
        super(Pairwise, self).__init__()
        self.context_size = context_size
        self.pad = context_size // 2
        self.conv = torch.nn.Conv2d(
            self.context_size * self.context_size,
            output_channel * 2,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        self.conv1 = torch.nn.Conv2d(
            output_channel * 2,
            output_channel,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        self.conv2 = torch.nn.Conv2d(
            output_channel,
            output_channel,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        if use_cuda:
            self.conv = self.conv.cuda()
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()

    def self_similarity(self, feature_normalized):
        b, c, h, w = feature_normalized.size()
        feature_pad = F.pad(
            feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0
        )
        output = torch.zeros(
            [self.context_size * self.context_size, b, h, w],
            dtype=feature_normalized.dtype,
            requires_grad=feature_normalized.requires_grad,
        )
        if feature_normalized.is_cuda:
            output = output.cuda(feature_normalized.get_device())
        for c in range(self.context_size):
            for r in range(self.context_size):
                output[c * self.context_size + r] = (
                    feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized
                ).sum(1)

        output = output.transpose(0, 1).contiguous()
        return output

    def forward(self, feature):
        feature_normalized = F.normalize(feature, p=2, dim=1)
        ss = self.self_similarity(feature_normalized)

        ss1 = F.relu(self.conv(ss))
        ss2 = F.relu(self.conv1(ss1))
        output = torch.cat((ss, ss1, ss2), 1)
        return output


def CreateCon4D(k1, k2, channels):
    num_layers = len(channels)
    nn_modules = list()
    for i in range(1, num_layers):
        nn_modules.append(
            Conv4d(
                in_channels=channels[i - 1],
                out_channels=channels[i],
                kernel_size=[k1, k1, k2, k2],
                bias=True,
            )
        )
        nn_modules.append(nn.ReLU(inplace=True))
    conv = nn.Sequential(*nn_modules)
    return conv


class NeighConsensus(torch.nn.Module):
    def __init__(
        self,
        use_cuda=True,
        kernel_sizes=[3, 3, 3],
        channels=[1, 10, 10, 1],
        symmetric_mode=True,
    ):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.conv = CreateCon4D(kernel_sizes[0], kernel_sizes[0], channels)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


class NonIsotropicNCA(torch.nn.Module):
    def __init__(self, use_cuda=True, channels=[1, 16, 16, 1], symmetric_mode=True):
        super(NonIsotropicNCA, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.conv0 = CreateCon4D(3, 5, [1, 8])
        self.conv1 = CreateCon4D(5, 5, [1, 8])
        self.conv2 = CreateCon4D(5, 5, [16, 16, 1])
        if use_cuda:
            self.conv0.cuda()
            self.conv1.cuda()
            self.conv2.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x0 = self.conv0(x) + self.conv0(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x1 = self.conv1(x) + self.conv1(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x) + self.conv2(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x0 = self.conv0(x)
            x1 = self.conv1(x)
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x)
        return x


class NonIsotropicNCB(torch.nn.Module):
    def __init__(self, use_cuda=True, channels=[1, 16, 16, 1], symmetric_mode=True):
        super(NonIsotropicNCB, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.conv0 = CreateCon4D(5, 5, [1, 16])
        self.conv10 = CreateCon4D(3, 5, [16, 8])
        self.conv11 = CreateCon4D(5, 5, [16, 8])
        self.conv2 = CreateCon4D(5, 5, [16, 1])
        if use_cuda:
            self.conv0.cuda()
            self.conv10.cuda()
            self.conv11.cuda()
            self.conv2.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv0(x) + self.conv0(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x0 = self.conv10(x) + self.conv10(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x1 = self.conv11(x) + self.conv11(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x) + self.conv2(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv0(x)
            x0 = self.conv10(x)
            x1 = self.conv11(x)
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x)
            # because of the ReLU layers in between linear layers,
        return x


class NonIsotropicNCC(torch.nn.Module):
    def __init__(self, use_cuda=True, channels=[1, 16, 16, 1], symmetric_mode=True):
        super(NonIsotropicNCC, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.conv00 = CreateCon4D(3, 5, [1, 8])
        self.conv01 = CreateCon4D(5, 5, [1, 8])
        self.conv10 = CreateCon4D(3, 5, [16, 8])
        self.conv11 = CreateCon4D(5, 5, [16, 8])
        self.conv2 = CreateCon4D(5, 5, [16, 1])
        if use_cuda:
            self.conv00.cuda()
            self.conv01.cuda()
            self.conv10.cuda()
            self.conv11.cuda()
            self.conv2.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x0 = self.conv00(x) + self.conv00(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x1 = self.conv01(x) + self.conv01(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x = torch.cat((x0, x1), 1)
            x0 = self.conv10(x) + self.conv10(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x1 = self.conv11(x) + self.conv11(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x) + self.conv2(x.permute(0, 1, 4, 5, 2, 3)).permute(
                0, 1, 4, 5, 2, 3
            )
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x0 = self.conv00(x)
            x1 = self.conv01(x)
            x = torch.cat((x0, x1), 1)
            x0 = self.conv10(x)
            x1 = self.conv11(x)
            x = torch.cat((x0, x1), 1)
            x = self.conv2(x)
        return x


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (
        corr4d_A * corr4d_B
    )  # parenthesis are important for symmetric output

    return corr4d


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(
                        corr4d_hres[
                            :, 0, i::k_size, j::k_size, k::k_size, l::k_size
                        ].unsqueeze(0)
                    )
    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


class ImMatchNet(nn.Module):
    def __init__(
        self,
        feature_extraction_cnn="resnet101",
        feature_extraction_last_layer="",
        feature_extraction_model_file=None,
        return_correlation=False,
        ncons_kernel_sizes=[3, 3, 3],
        ncons_channels=[1, 10, 10, 1],
        normalize_features=True,
        train_fe=False,
        use_cuda=True,
        relocalization_k_size=0,
        half_precision=False,
        checkpoint=None,
        pss=True,
        noniso=1,
    ):

        super(ImMatchNet, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.return_correlation = return_correlation
        self.relocalization_k_size = relocalization_k_size
        self.half_precision = half_precision
        self.pss = pss
        self.noniso = noniso

        self.FeatureExtraction = FeatureExtraction(
            train_fe=train_fe,
            feature_extraction_cnn=feature_extraction_cnn,
            feature_extraction_model_file=feature_extraction_model_file,
            last_layer=feature_extraction_last_layer,
            normalization=normalize_features,
            use_cuda=self.use_cuda,
        )

        self.FeatureCorrelation = FeatureCorrelation(normalization=False)

        if self.noniso == 0:
            self.NeighConsensus = NeighConsensus(
                use_cuda=self.use_cuda,
                kernel_sizes=ncons_kernel_sizes,
                channels=ncons_channels,
            )
        elif self.noniso == 1:
            self.NeighConsensus = NonIsotropicNCA(
                use_cuda=self.use_cuda, channels=ncons_channels
            )
        elif self.noniso == 2:
            self.NeighConsensus = NonIsotropicNCB(
                use_cuda=self.use_cuda, channels=ncons_channels
            )
        elif self.noniso == 3:
            self.NeighConsensus = NonIsotropicNCC(
                use_cuda=self.use_cuda, channels=ncons_channels
            )

        if self.pss == 1:
            self.SS = Pairwise(
                ncons_kernel_sizes[0], output_channel=32, use_cuda=self.use_cuda
            )
        elif self.pss == 2:
            self.SS = SpatialContextNet(
                ncons_kernel_sizes[0], output_channel=256, use_cuda=self.use_cuda
            )
        # Load weights
        if checkpoint is not None and checkpoint is not "":
            print("Copying weights...")
            for name, param in self.FeatureExtraction.state_dict().items():
                if "num_batches_tracked" not in name:
                    self.FeatureExtraction.state_dict()[name].copy_(
                        checkpoint["state_dict"]["module.FeatureExtraction." + name]
                    )
            for name, param in self.NeighConsensus.state_dict().items():
                self.NeighConsensus.state_dict()[name].copy_(
                    checkpoint["state_dict"]["module.NeighConsensus." + name]
                )
            if self.pss > 0:
                for name, param in self.SS.state_dict().items():
                    self.SS.state_dict()[name].copy_(
                        checkpoint["state_dict"]["module.SS." + name]
                    )

            print("Done!")

        self.FeatureExtraction.eval()

        if self.half_precision:
            for p in self.NeighConsensus.parameters():
                p.data = p.data.half()
            for l in self.NeighConsensus.conv:
                if isinstance(l, Conv4d):
                    l.use_half = True

    def ncnet(self, corr4d):
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d = MutualMatching(corr4d)
        return corr4d

    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch):
        """
        Arguments:
            tnf_batch [dict]: source_image B x 3 x H x W image 256 x 256
                              target_image B x 3 x H x W image 
        Return: 
            corr4d [Tensor float]: B x 1 x 16 x 16 x 16 x 16
        """
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch["source_image"])  # B x C x 16 x 16
        feature_B = self.FeatureExtraction(tnf_batch["target_image"])

        if self.half_precision:
            feature_A = feature_A.half()
            feature_B = feature_B.half()
        # feature correlation
        corr4d = self.FeatureCorrelation(
            feature_A, feature_B
        )  # B x 1 x 16 x 16 x 16 x 16
        # do 4d maxpooling for relocalization
        if self.relocalization_k_size > 1:
            corr4d, max_i, max_j, max_k, max_l = maxpool4d(
                corr4d, k_size=self.relocalization_k_size
            )

        # run match processing model
        corr4d = self.ncnet(corr4d)

        # pss is pairwise term
        if self.pss > 0:
            selfsimilarity_A = self.SS(feature_A)
            selfsimilarity_B = self.SS(feature_B)
            corr4d_s = self.FeatureCorrelation(selfsimilarity_A, selfsimilarity_B)
            # do 4d maxpooling for relocalization
            if self.relocalization_k_size > 1:
                corr4d_s, max_i_s, max_j_s, max_k_s, max_l_s = maxpool4d(
                    corr4d_s, k_size=self.relocalization_k_size
                )

            # run match processing model
            corr4d_s = self.ncnet(corr4d_s)
            corr4d = 0.5 * corr4d + 0.5 * corr4d_s

        if self.relocalization_k_size > 1:
            delta4d = (max_i, max_j, max_k, max_l)
            delta4d_s = (max_i_s, max_j_s, max_k_s, max_l_s)

            return (corr4d, delta4d, delta4d_s)
        else:
            return corr4d
