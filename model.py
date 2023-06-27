from torchvision.models import regnet_y_16gf, efficientnet_b6
import torch.nn as nn



class My_regnet(nn.Module):
    def __init__(self, num_classes):
        super(My_regnet, self).__init__()
        self.model = regnet_y_16gf(pretrained=True)
        self.model.fc = nn.Linear(3024, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class My_efficientnet(nn.Module):
    def __init__(self, num_classes):
        super(My_efficientnet, self).__init__()
        self.model = efficientnet_b6(pretrained=True)
        self.model.classifier[1] = nn.Linear(2304, num_classes, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x
