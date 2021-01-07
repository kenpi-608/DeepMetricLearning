from tqdm import tqdm
from torch import nn, Tensor

from circle_loss import convert_label_to_similarity


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        # feature = self.feature_extractor(input).mean(dim=[2, 3])
        feature = self.feature_extractor(input).view(-1, 32)
        return nn.functional.normalize(feature)


def train(model, criterion, optimizer, epoch, loader, device):
    print("Training... Epoch = %d" % epoch)
    for img, label in tqdm(loader):
        img, label = img.to(device), label.to(device)
        model.zero_grad()
        pred = model(img)
        loss = criterion(*convert_label_to_similarity(pred, label))
        loss.backward()
        optimizer.step()
