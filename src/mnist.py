from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_mnist_loader(is_train: bool, batch_size: int) -> DataLoader:
    trainset = datasets.MNIST(
                                root="../data",
                                train=is_train,
                                transform=ToTensor(),
                                download=True
                              )
    return DataLoader(trainset, batch_size=batch_size, shuffle=is_train)
