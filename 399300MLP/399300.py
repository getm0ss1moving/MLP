from tqdm import tqdm
import random
import time

import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error 
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

class model(nn.Module):
    def __init__(self,input_dim)->None:
        super(model,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self ,x):
        return self.network(x).squeeze(-1)
print("读取内容中")
train_df = pd.read_csv('problem1/399300/399300MLP/399300.csv')
print("数据读取完成")
idx_fea = ('open','high','low','pre_close','volume','turnover_ratio','amount')
out_fea = 'OT' 
print("正在检查设备...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"将使用设备: {device}")
train_df['date'] = pd.to_datetime(train_df['date'])
X_list = []
y_list = []
print("正在构建特征和标签...")
for i in tqdm(range(20,len(train_df))):
    window = train_df.iloc[i-20:i]
    features = window[list(idx_fea)].values
    x_input = features.flatten()
    y_target = train_df.loc[i, out_fea]
    X_list.append(x_input)
    y_list.append(y_target)
X = np.stack(X_list)
y = np.stack(y_list).reshape(-1,1)
# 数据集划分和标准化
total_samples = len(X)
train_end_idx = int(total_samples * 0.7)
val_end_idx = int(total_samples * 0.8)
X_train_raw = X[:train_end_idx]
y_train_raw = y[:train_end_idx]
X_val_raw = X[train_end_idx:val_end_idx]
y_val_raw = y[train_end_idx:val_end_idx]
X_test_raw = X[val_end_idx:]
y_test = y[val_end_idx:] # y_test是原始值
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)
# 转换为Tensor 
X_train_tensor = torch.tensor(X_train,dtype = torch.float32).to(device)
X_val_tensor = torch.tensor(X_val,dtype = torch.float32).to(device)
X_test_tensor = torch.tensor(X_test,dtype = torch.float32).to(device)

y_train_tensor = torch.tensor(y_train_raw,dtype = torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_raw,dtype = torch.float32).to(device)
# 创建DataLoader
train_data = TensorDataset(X_train_tensor,y_train_tensor)
val_data = TensorDataset(X_val_tensor,y_val_tensor)
train_loader = DataLoader(train_data,batch_size = 64,shuffle = True)
val_loader = DataLoader(val_data,batch_size = 64,shuffle = False)
# 模型、优化器、损失函数
input_dim = X_train.shape[1]
mymodel = model(input_dim).to(device)
optimizer = optim.Adam(mymodel.parameters(),lr  = 0.001)
loss_fn = nn.MSELoss()
# 训练循环
epochs = 500
best_loss = float('inf')
patience = 20
cnt = 0
print("\n开始模型训练...")
for epoch in range (epochs):
    mymodel.train()
    train_loss = 0.0
    for X_batch,y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = mymodel(X_batch)
        loss = loss_fn(y_pred,y_batch.squeeze(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    mymodel.eval()
    val_loss = 0.0
    with torch.no_grad():
         for X_batch, y_batch in val_loader:
            y_pred = mymodel(X_batch)
            loss = loss_fn(y_pred, y_batch.squeeze(-1))
            val_loss += loss.item() 
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs} | 训练 Loss: {avg_train_loss:.6f} | 验证 Loss: {avg_val_loss:.6f}")
    
    if avg_val_loss < best_loss:
        cnt = 0
        best_loss = avg_val_loss
        torch.save(mymodel.state_dict(),'best_model.pth')
    else:
        cnt += 1
        if cnt >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break
print("\n加载最佳模型进行评估...")
predictor = model(input_dim).to(device)
predictor.load_state_dict(torch.load('best_model.pth'))
predictor.eval()
# 获取预测结果，y未标准化，此结果即为最终预测值
predictions = []
with torch.no_grad():
    predictions = predictor(X_test_tensor).cpu().numpy()
# 准备真实值和预测值
actual = y_test.flatten()
predictions = predictions.flatten()
# 计算五项指标 
print(f"\n--- {out_fea} 预测评估指标 ---")
# MAE 
mae = mean_absolute_error(actual, predictions)
print(f"平均绝对误差 (MAE): {mae:.4f}")
# RMSE
rmse = np.sqrt(mean_squared_error(actual, predictions))
print(f"均方根误差 (RMSE): {rmse:.4f}")
# ACC
if len(y_val_raw) > 0:
    last_val_target = y_val_raw[-1]
else:
    last_val_target = train_df.loc[val_end_idx-1, out_fea]
# 拼接，进行列操作
previous_day_values = np.concatenate([last_val_target, y_test[:-1].flatten()])
actual_change = actual - previous_day_values
predicted_change = predictions - previous_day_values
correct_direction = (actual_change * predicted_change) >= 0
acc = np.mean(correct_direction)
print(f"方向预测准确性 (ACC): {acc:.2%}")
# 如果预测第二天会上涨，全买
position = np.where(predictions > previous_day_values, 1, 0)
# 计算每日的实际收益率
actual_returns = np.divide(actual_change, previous_day_values, 
                           out=np.zeros_like(actual_change, dtype=float), 
                           where=previous_day_values!=0)
# 策略每日收益率 = 实际每日收益率 * 头寸信号（这里只有1和0）
strategy_returns = actual_returns * position
# CR 
cumulative_return = (np.prod(1 + strategy_returns) - 1)
print(f"策略累计回报率 (CR): {cumulative_return:.2%}")
# Sharpe Ratio
std_dev = np.std(strategy_returns)
if std_dev > 0:
    sharpe_ratio = np.sqrt(252) * (np.mean(strategy_returns) / std_dev)
else:
    sharpe_ratio = 0.0
print(f"策略年化夏普比率: {sharpe_ratio:.4f}")
# 保存
print("\n正在生成预测结果文件...")
test_dates = train_df.iloc[val_end_idx + 20:].reset_index(drop=True)['date']
results_df = pd.DataFrame({
    'Date': test_dates,
    '真实值': actual,
    '预测值': predictions
})
# 保存到CSV文件
output_filename = 'prediction_results.csv'
results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\n完整预测结果已保存至文件: {output_filename}")
