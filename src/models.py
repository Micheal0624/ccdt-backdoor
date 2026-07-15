import torch
import torch.nn as nn
import torchvision.models as tv_models


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        model = tv_models.resnet18(weights=None)
    except TypeError:
        model = tv_models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class VGG11Small(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()

    if name == "resnet18":
        return build_resnet18(num_classes)

    if name == "vgg11":
        return VGG11Small(num_classes)

    raise ValueError(f"Unknown model: {model_name}")
