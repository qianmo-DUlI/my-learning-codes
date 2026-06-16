# ====================== 1. 导入所有必要的库 ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ====================== 2. 设备配置 ======================
# 自动选择GPU（如果可用），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ====================== 3. 定义神经网络模型 ======================
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ====================== 4. 数据准备与加载 ======================
# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# ====================== 5. 初始化模型、损失函数和优化器 ======================
model = SimpleNN(input_size=784, hidden_size=128, output_size=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ====================== 6. 训练模型 ======================
num_epochs = 5
total_step = len(train_loader)
train_losses = []  # 用于记录训练损失

print("\n开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到设备
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        running_loss += loss.item()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

    # 计算每个epoch的平均损失
    epoch_loss = running_loss / total_step
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}] 平均损失: {epoch_loss:.4f}\n")

# ====================== 7. 测试模型 ======================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")

# ====================== 8. 可视化训练损失 ======================
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.title('训练损失变化')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.grid(True)
plt.show()

# ====================== 9. 保存模型 ======================
torch.save(model.state_dict(), "mnist_final_model.pth")
print("\n模型已保存为 mnist_final_model.pth")