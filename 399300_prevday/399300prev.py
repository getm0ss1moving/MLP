import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv('python/pythonpro1/399300.csv')
data['date'] = pd.to_datetime(data['date'])
predictions = []
for idx, row in data.iterrows():
    earlier_data = data[data['date'] < row['date']]
    if earlier_data.empty:
        prediction = data['OT'].mean()
    else:
        closest_record = earlier_data.loc[earlier_data['date'].idxmax()]
        prediction = closest_record['OT']
    predictions.append({
        'date': row['date'].strftime('%Y-%m-%d'), 
        'actual': row['OT'],
        'prediction': prediction
    })
predictions_df = pd.DataFrame(predictions)
print("\n模型性能评估指标")
split_index = int(len(predictions_df) * 0.8)
y_true = predictions_df['actual'].iloc[split_index:]
y_pred = predictions_df['prediction'].iloc[split_index:]
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
# 0 = 会涨
actual_direction = np.sign(y_true.diff().dropna())
predicted_direction = np.sign(y_pred - y_true.shift(1))
predicted_direction[predicted_direction == 0] = 1
predicted_direction = predicted_direction.dropna()
correct_directions = (actual_direction.values == predicted_direction.values)
acc = np.mean(correct_directions) if len(correct_directions) > 0 else 0
print(f"方向准确率 (ACC): {acc:.4f} ({acc:.2%})")
signal = np.where(y_pred >= y_true.shift(1), 1, 0)
daily_returns = y_true.pct_change()
strategy_returns = daily_returns * pd.Series(signal, index=daily_returns.index).shift(1)
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
predictions_df.to_csv('python/pythonpro1/399300_prev_predicitions.csv', index=False)