import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
from model import ResNeXt
from mydataset import get_dataloaders

def train():
    # ---------- 基本配置 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "data"                # 修改为你的数据路径
    batch_size = 32
    epochs = 80
    lr = 0.01

    # ---------- 数据加载 ----------
    train_loader, val_loader, _, classes = get_dataloaders(data_root, batch_size)
    num_classes = len(classes)

    # ---------- 模型、损失、优化器 ----------
    model = ResNeXt(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # 余弦退火调整学习率

    # ---------- 保存目录 ----------
    os.makedirs("model-out", exist_ok=True)
    log_file = open("train_log.txt", "w")

    best_acc = 0.0  # 记录最佳验证准确率

    for epoch in range(1, epochs+1):
        # ================= 训练 =================
        model.train()
        train_loss = 0.0
        train_correct = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        scheduler.step()  # 每个epoch后更新学习率
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # ================= 验证 =================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # ---------- 日志与保存 ----------
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        log_file.write(log_msg + "\n")
        print(log_msg)

        # 保存最佳模型（基于验证集准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model-out/best.pth")
        torch.save(model.state_dict(), "model-out/last.pth")

    log_file.close()
    print("Training complete. Best val accuracy: {:.4f}".format(best_acc))

if __name__ == "__main__":
    train()