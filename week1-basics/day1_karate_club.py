import torch
import torch.nn.functional as F
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv

dataset = KarateClub()
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

torch.manual_seed(42)
model = GCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(200):
    model.train(); opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward(); opt.step()
model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc = (pred == data.y).float().mean()
print(f"训练 200 轮 | 最终 loss: {loss.item():.4f} | 全图预测准确率: {acc:.3f}")
test_acc = (pred[~data.train_mask] == data.y[~data.train_mask]).float().mean()
print(f"4 个未标注节点（测试集）准确率: {test_acc:.3f}")
