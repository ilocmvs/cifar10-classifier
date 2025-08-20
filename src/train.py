import torch
import torchvision
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_loaders(data_dir="data", batch_size=128):
    transform_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return (
        DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        ),
        DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        ),
    )


def get_model(num_classes=10):
    model = torchvision.models.resnet18(num_classes=num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc="train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="eval"):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_loaders()
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    epochs = 1  # bump later
    for ep in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc = eval_acc(model, test_loader, device)
        print(f"epoch {ep}: loss={loss:.4f}  acc={acc*100:.2f}%")


if __name__ == "__main__":
    main()
