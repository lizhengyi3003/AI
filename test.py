import torch
from model import ResNeXt
from mydataset import get_dataloaders

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = get_dataloaders("data")

    model = ResNeXt(num_classes=len(classes)).to(device)
    # 加载最佳权重
    model.load_state_dict(torch.load("model-out/best.pth", map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    test()