import torch
import torch.nn as nn
from torchvision import models


class Dense(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            # nn.Dropout(0.5),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.dense(x)

        return out


class Linear(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):

        return self.fc(x)


class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        plane = 512
        if model_name == "resnet18":
            backbone = nn.Sequential(
                *list(models.resnet18(pretrained=pretrained).children())[:-2]
            )
            plane = 512 * 1 * 1
        elif model_name == "resnet50":
            backbone = nn.Sequential(
                *list(models.resnet50(pretrained=pretrained).children())[:-2]
            )
            plane = 2048 * 1 * 1
        elif model_name == "resnet101":
            backbone = nn.Sequential(
                *list(models.resnet101(pretrained=pretrained).children())[:-2]
            )
            plane = 2048 * 1 * 1
        elif model_name == 's101':
            torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
            # load pretrained models, using ResNeSt-50 as an example
            model = torch.hub.load("zhanghang1989/ResNeSt", "resnest101", pretrained=pretrained)
            backbone = nn.Sequential(*list(model.children())[:-2])
        elif model_name == "vgg16bn":
            backbone = nn.Sequential(
                *list(models.vgg16(pretrained=pretrained).children())[:-2]
            )
            plane = 512 * 1 * 1
        else:
            backbone = None

        self.backbone = backbone
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        feat = self.backbone(x)
        out = self.maxpool(feat)
        out = out.view((out.size(0), -1))

        return feat, out


class BaseModel_scratch(nn.Module):
    def __init__(self, model_name, eps=3, num_classes=200, init_weights=True):
        super().__init__()
        if model_name == "vgg16bn":
            backbone = nn.Sequential(
                *list(models.vgg16_bn(pretrained=True).features.children())[:-4]
            )
            last_conv = nn.Sequential(
                nn.Conv2d(512, num_classes * eps, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_classes * eps),
                nn.ReLU(True),
                nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
            )
        else:
            backbone = None
            last_conv = None

        self.backbone = backbone
        self.last_conv = last_conv

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.last_conv(feat)

        out = self.maxpool(feat)
        out = out.view(out.size(0), -1)

        return feat, out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = BaseModel_scratch("vgg16bn", 3, 1000, False)
    inp = torch.randn((3, 3, 224, 224))
    a, b = model(inp)
    print(a.size())
    print(b.size())
