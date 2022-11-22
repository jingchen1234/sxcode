import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152





class ChannelAttentionModule(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, inplanes):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(inplanes)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        if pretrained_path:
            backbone.load_state_dict(torch.load(pretrained_path))


        self.early_extractor = nn.Sequential(*list(backbone.children())[:3])
        self.early_extractor_ = nn.Sequential(*list(backbone.children())[3:5])
        self.later_extractor_ = nn.Sequential(*list(backbone.children())[5:6])
        self.later_extractor = nn.Sequential(*list(backbone.children())[6:7])

        conv4_block1 = self.later_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):

        x = self.early_extractor(x)
        # print(x.shape)
        x_ = self.early_extractor_(x)
        # print(x_.shape)
        out_ = self.later_extractor_(x_)
        # print(out_.shape)
        out = self.later_extractor(out_)
        # print(out.shape)
        return  x,x_,out_,out

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=2048, output_stride=16,):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # self.expansion = expansion
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.global_avg_pool2 = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(inplace=True))
        self.global_avg_pool3 = nn.Sequential(nn.AdaptiveAvgPool2d((3, 3)),
                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True))
        self.global_avg_pool4 = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)),
                                              nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True))
        self.maxpool         =nn.Sequential(nn.MaxPool2d(1,1),
                                            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        # self.conv1 = nn.Conv2d(3072, 256, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)
        self._init_weight()
        self.cbam = CBAM(256)

    def forward(self, x):
        # print(x.shape)
        x1 = self.aspp1(x)
        x1 = self.cbam(x1)
        x2 = self.aspp2(x)
        x2 = self.cbam(x2)
        x3 = self.aspp3(x)
        x3 = self.cbam(x3)
        x4 = self.aspp4(x)
        x4 = self.cbam(x4)
        x5 = self.global_avg_pool1(x)
        x5 = self.cbam(x5)
        x6 = self.global_avg_pool2(x)
        x6 = self.cbam(x6)
        x7 = self.global_avg_pool3(x)
        x7 = self.cbam(x7)
        x8 = self.global_avg_pool4(x)
        x8 = self.cbam(x8)
        x9 = self.maxpool(x)
        # print(x9.shape)
        x9 = self.cbam(x9)


        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x6 = F.interpolate(x6, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x7 = F.interpolate(x7, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x8 = F.interpolate(x8, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x,x1, x2, x3, x4, x5,x6,x7,x8,x9), dim=1)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




class convlow(nn.Module):
    def __init__(self, inplanes, planes,stride):
        super(convlow, self).__init__()
        self.convlow=nn.Sequential(nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU(inplace=True))
    def forward(self, x):
       x_out=self.convlow(x)
       return x_out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup,inplanes,planes,stride, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.convlow=convlow(inplanes,planes,stride)
        self.c1     =nn.Conv2d(inp,oup,kernel_size=1,bias=False)

    def forward(self, x,x_low):
        identity = x

        x_low=self.convlow(x_low)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()


        out = a_w * a_h*x_low
        # out =out*x_low
        identity=self.c1(identity)
        out=out+identity
        return out






class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)










class ChannelAttentionModule1(nn.Module):
    def __init__(self, inplanes, ratio=16):
        super(ChannelAttentionModule1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(inplanes // ratio, inplanes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule1(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule1, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM1(nn.Module):
    def __init__(self, inplanes):
        super(CBAM1, self).__init__()
        self.channel_attention1 = ChannelAttentionModule1(inplanes)
        self.spatial_attention1 = SpatialAttentionModule1()

        self.simam=simam_module()

    def forward(self, x):
        x_out =self.simam(x)

        out = self.channel_attention1(x) * x
        out1=out+x_out
        out2=self.simam(out1)

        out3 = self.spatial_attention1(out1) * out1

        out4  =out2+out3
        return out4






class Our(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = ResNet('resnet50', None)
        self.aspp = ASPP(inplanes=self.backbone.final_out_channels)
        # self.decoder = Decoder(self.num_classes, self.backbone.low_level_inplanes)


        self.last_conv1= nn.Sequential(nn.Conv2d(1056, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(512),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(512),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.1),
                                               nn.Conv2d(512, num_classes, kernel_size=1, stride=1))

        self.up1=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.c3 =nn.Sequential(nn.Conv2d(768, 144, 1, bias=False),
                              nn.BatchNorm2d(144),
                              nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(512, 96, 1, bias=False),
                              nn.BatchNorm2d(96),
                              nn.ReLU())
        self.c1 = nn.Sequential(nn.Conv2d(320, 48, 1, bias=False),
                              nn.BatchNorm2d(48),
                              nn.ReLU())


        self.conv1 = nn.Conv2d(3328, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.ca3=CoordAtt(1024,512,512,512,1)
        self.ca2=CoordAtt(512,256,256,256,2)
        self.ca1=CoordAtt(256,64,64,64,2)

        self.cbam3 =CBAM1(144)
        self.cbam2 = CBAM1(96)
        self.cbam1 = CBAM1(48)

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        # print(self.backbone)
        a,b,c,d = self.backbone(imgs)
        x = self.aspp(d)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x =self.dropout(x)

        x_ca3=self.ca3(d,c)
        x_ca2=self.ca2(c,b)
        x_ca1=self.ca1(b,a)
        # print(x_ca3.shape)
        # print(x_ca2.shape)
        # print(x_ca1.shape)



        a3 = torch.cat([x, x_ca3], dim=1)
        a3_out = self.c3(a3)
        a3_out =self.cbam3(a3_out)
        a3_out =torch.cat([x,a3_out],dim=1)
        # x2 = self.up1(x)


        a2 = torch.cat([x, x_ca2], dim=1)
        a2_out = self.c2(a2)
        a2_out = self.cbam2(a2_out)
        a2_out = torch.cat([x, a2_out], dim=1)


        x1 = self.up1(x)
        a1 = torch.cat([x1, x_ca1], dim=1)
        a1_out = self.c1(a1)
        a1_out = self.cbam1(a1_out)
        a1_out = torch.cat([x1, a1_out], dim=1)


        a2_out=self.up1(a2_out)
        a3_out = self.up1(a3_out)
        x_cat =torch.cat([a1_out,a2_out,a3_out],dim=1)


        # x3_out = self.last_conv3(a3_out)
        # x2_out = self.last_conv2(a2_out)
        x = self.last_conv1(x_cat)


        # x3_out = F.interpolate(x3_out, size=x1_out.size()[2:], mode='bilinear', align_corners=True)
        # x2_out = F.interpolate(x2_out, size=x1_out.size()[2:], mode='bilinear', align_corners=True)



        # x=x2_out+x3_out+x1_out

        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)
        return outputs

if __name__ == '__main__':
    from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model =Our(num_classes=3)
    # print(model)
    batch = torch.FloatTensor(1, 3, 224, 224)

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)

    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))