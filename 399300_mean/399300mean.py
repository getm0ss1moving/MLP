import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
data = pd.read_csv('problem1/399300/399300_mean/399300.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)
predictions = []
window_size = 10 # 明确定义窗口大小

print("正在使用 for 循环和滑动窗口生成预测...")
for i in range(len(data)):
    
    # 确定当前窗口的起始索引
    # 使用 max(0, ...) 来处理开头不足10个数据点的情况
    start_index = max(0, i - window_size)
    
    # 提取从 start_index 到 i-1 的数据窗口
    # 注意：切片 data[start_index:i] 不包含索引为 i 的行，正好是我们需要的历史数据
    data_window = data.iloc[start_index:i]
    if data_window.empty:
        # 这种情况只会在 i=0 时发生
        # 第一天没有历史数据，可以简单地用自己的值填充，以避免后续计算出问题
        prediction = data.loc[i, 'OT']
    else:
        # 使用窗口内数据的均值作为预测
        prediction = data_window['OT'].mean()
    predictions.append({
        'date': data.loc[i, 'date'].strftime('%Y-%m-%d'), 
        'actual': data.loc[i, 'OT'],
        'prediction': prediction
    })

predictions_df = pd.DataFrame(predictions)


print("\n--- 模型性能评估指标 (仅在最后20%的数据上计算) ---")

# 计算分割点
split_index = int(len(predictions_df) * 0.8)

# 直接分割出测试集所需的 y_true 和 y_pred
y_true_test = predictions_df['actual'].iloc[split_index:]
y_pred_test = predictions_df['prediction'].iloc[split_index:]

if y_true_test.empty:
    print("测试集为空，无法计算指标。")
else:
    # RMSE 和 MAE
    rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
    mae = mean_absolute_error(y_true_test, y_pred_test)
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")

    # 获取测试集 "前一天" 的真实价格序列
    last_train_actual = predictions_df['actual'].iloc[split_index - 1]
    previous_day_actuals = pd.concat([pd.Series([last_train_actual]), y_true_test.iloc[:-1]], ignore_index=True)
    previous_day_actuals.index = y_true_test.index
    # ACC: 方向准确率
    actual_direction = np.sign(y_true_test - previous_day_actuals)
    predicted_direction = np.sign(y_pred_test - previous_day_actuals)
    acc = np.mean(actual_direction.values == predicted_direction.values) if len(actual_direction) > 0 else 0
    print(f"方向准确率 (ACC): {acc:.4f} ({acc:.2%})")
    
    # CR 和 Sharpe Ratio
    signal = np.where(y_pred_test > previous_day_actuals, 1, 0)
    daily_returns = y_true_test.pct_change()
    strategy_returns = daily_returns * pd.Series(signal, index=y_true_test.index)
    strategy_returns = strategy_returns.fillna(0)
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    cr = cumulative_returns.iloc[-1] - 1
    print(f"累计回报率 (CR): {cr:.4f} ({cr:.2%})")

    if strategy_returns.std() > 0:
        daily_sharpe = strategy_returns.mean() / strategy_returns.std()
        sharpe_ratio = np.sqrt(252) * daily_sharpe
    else:
        sharpe_ratio = 0.0
    print(f"年化夏普比率 (Sharpe Ratio): {sharpe_ratio:.4f}")

# --- 4. 保存文件 (无变化) ---
predictions_df.to_csv('problem1/399300/399300_mean/399300_mean_predicitions.csv', index=False)
