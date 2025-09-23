from torch import nn
from torchvision.models import resnet, efficientnet_b0

class AgeEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.LazyLinear(1)


    def forward(self, x):
        y = self.model(x)
        y = self.fc(y.flatten(1))
        return y
