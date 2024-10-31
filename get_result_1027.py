import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random

# 加载数据
pred1 = np.load("bone_3d_2_B.npy")
pred2 = np.load("bone_3d_B.npy")
pred3 = np.load("joint_3d_2_B.npy")
pred4 = np.load("joint_3d_B.npy")
pred5 = np.load("1029B.npy")

# true_label = np.load("test_label_A.npy")

# Top 3 类别权重
top3_weights = [0.44677, 0.301245, 0.252]  # Top 3 类别权重

# 正确预测的平均 Top 1 置信度（假设已知）
correct_avg_confidences = {
    "bone_3d_1_A": 0.8549,
    "bone_3d_A": 0.8416,
    "joint_3d_1_A": 0.8487,
    "joint_3d_A": 0.8363,
    "TE-GCN-A-70": 17.8896
}

# 模型列表和名称
models = [pred1, pred2, pred3, pred4, pred5]
model_names = ["bone_3d_1_A", "bone_3d_A", "joint_3d_1_A", "joint_3d_A", "TE-GCN-A-70"]

# 调整置信度函数：按条件处理 Top 1 或按 Top 3 权重处理
def adjust_confidences(pred, correct_avg, top3_weights):
    adjusted_pred = np.zeros_like(pred)  # 初始化调整后的数组

    for i in range(pred.shape[0]):
        # 获取当前样本的 Top 1 类别索引和对应的置信度
        top1_index = np.argmax(pred[i])
        top1_confidence = pred[i][top1_index]


        # 按 Top 3 权重处理
        sorted_indices = np.argsort(pred[i])[::-1][:3]
        for j, idx in enumerate(sorted_indices):
            adjusted_pred[i][idx] = top3_weights[j]

    return adjusted_pred

# 计算加权融合后的预测结果
def weighted_fusion(models, model_weights, top3_weights):
    adjusted_preds = []

    # 对每个模型的预测结果进行置信度调整
    for i, model in enumerate(models):
        model_name = model_names[i]
        correct_avg = correct_avg_confidences[model_name]
        adjusted_pred = adjust_confidences(model, correct_avg, top3_weights)
        adjusted_preds.append(adjusted_pred)

    # 根据模型权重进行加权融合
    ensemble_pred = sum(model_weights[i] * adjusted_preds[i] for i in range(len(models)))

    return ensemble_pred

# 评估函数：根据模型权重计算准确率
def evaluate_accuracy(model_weights):
    ensemble_pred = weighted_fusion(models, model_weights, top3_weights)
    final_prediction = np.argmax(ensemble_pred, axis=1)
    accuracy = accuracy_score(true_label, final_prediction)
    return accuracy

# 退火算法：搜索最优模型权重
def simulated_annealing(init_weights, max_iter=1000, temp=10.0, cooling_rate=0.99):
    current_weights = init_weights
    current_accuracy = evaluate_accuracy(current_weights)
    best_weights = current_weights
    best_accuracy = current_accuracy

    for i in range(max_iter):
        # 随机扰动模型权重，确保其在 [0.01, 1.0] 范围内
        new_weights = [
            max(0.01, min(1.0, w + random.uniform(-0.05, 0.05)))
            for w in current_weights
        ]

        # 归一化权重，使其总和为 1.0
        weight_sum = sum(new_weights)
        new_weights = [w / weight_sum for w in new_weights]

        # 计算新权重的准确率
        new_accuracy = evaluate_accuracy(new_weights)

        # 接受新解的概率：根据温度和准确率差决定
        if new_accuracy > current_accuracy or random.random() < np.exp((new_accuracy - current_accuracy) / temp):
            current_weights = new_weights
            current_accuracy = new_accuracy

            # 更新最佳解
            if new_accuracy > best_accuracy:
                best_weights = new_weights
                best_accuracy = new_accuracy

        # 降低温度
        temp *= cooling_rate

        # 输出进度
        if i % 100 == 0:
            print(f"第 {i} 次迭代：当前最佳准确率={best_accuracy:.4f}，权重={best_weights}")

    return best_weights, best_accuracy

# 初始化权重
init_weights = [0.1, 0.22, 0.23, 0.2, 0.25]  # 初始值为均匀分配

# 运行退火算法搜索最优权重
# best_weights, best_accuracy = simulated_annealing(init_weights)
best_weights = init_weights
# best_accuracy = evaluate_accuracy(best_weights)
# 输出最优权重和最高准确率
print(f"\n最优模型权重：{best_weights}")
# print(f"对应的最高准确率：{best_accuracy:.4f}")

# 保存最优融合结果为 npy 文件
ensemble_pred = weighted_fusion(models, best_weights, top3_weights)
print(ensemble_pred.shape)
np.save("pred_1029_final_final.npy", ensemble_pred)

# 保存最优融合结果为 xlsx 文件
df = pd.DataFrame(ensemble_pred)
df.to_excel("pred_1029_final_final.xlsx", index=False)

print("最优融合预测结果已保存为 'optimized_ensemble_prediction.npy' 和 'optimized_ensemble_prediction.xlsx'")
