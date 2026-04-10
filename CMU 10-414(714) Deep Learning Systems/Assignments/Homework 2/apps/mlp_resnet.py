import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Residual(
        nn.Sequential(
          nn.Linear(dim, hidden_dim),
          norm(hidden_dim), 
          nn.ReLU(),
          nn.Dropout(drop_prob),
          nn.Linear(hidden_dim, dim),
          norm(dim)
        )
      ),
      nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
      modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    # 使用 * 拆包列表
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
      model.train()

    total_error = 0.0
    total_loss = 0.0
    total_samples = 0
    
    # 获取损失函数实例
    loss_fn = nn.SoftmaxLoss()
    flat = nn.Flatten()

    for batch in dataloader:
      X, y = batch

      # 前向传播
      logits = model(flat(X))
      loss = loss_fn(logits, y)

      # 统计误差 (Error count)
      # logits 是 (batch, classes), y 是 (batch,)
      # 我们需要找到 logits 中每一行最大值的索引作为预测值
      preds = np.argmax(logits.numpy(), axis=1)
      total_error += (preds != y.numpy()).sum() # 这样写比 np.sum 略快

      # 统计 Loss (注意要乘以 batch size，因为 SoftmaxLoss 返回的是平均值)
      # 必须使用 .data.numpy() 或 .numpy() 来断开计算图，否则内存会爆炸
      total_loss += loss.data.numpy() * X.shape[0]
      total_samples += X.shape[0]

      # 更新参数
      if opt is not None:
        opt.reset_grad() # 清空梯度
        loss.backward() # 反向传播计算梯度
        opt.step() # 更新参数

    return total_error/total_samples, total_loss/total_samples
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # 定义数据增强
    train_transforms = [
        ndl.data.RandomFlipHorizontal(),
        ndl.data.RandomCrop(padding=3)
    ]
    # 惯例：训练集需要 shuffle=True 以增加随机性，但测试集通常设为 shuffle=False
    # 虽然设为 True 不会改变准确率的结果，但在工业界和学术界，为了保证测试过程的确定性和可重复性，测试集一般不打乱
    # 训练集使用数据增强和 Shuffle, 但是使用数据增强测试不通过
    train_dataloader = ndl.data.DataLoader(
      ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")), batch_size, shuffle=True)
    # 测试集不使用增强，shuffle 设为 False
    test_dataloader = ndl.data.DataLoader(
      ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")), batch_size, shuffle=False)
    
    model = MLPResNet(784, hidden_dim=hidden_dim)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for i in range(epochs):
      train_error, train_loss = epoch(train_dataloader, model, opt)
      test_error, test_loss = epoch(test_dataloader, model)
      # print 每个 epoch 的结果，方便观察模型是否收敛
        # print(f"Epoch {i}: Train Loss {train_loss:.3f}, Test Err {test_error:.3f}")
    
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
