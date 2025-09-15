import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 定义所有分类标签
allLabels = ['AM', 'COMB', 'FM', 'ISRJ', 'MNJ', 'R_VGPO', 'RGPO', 'RMT', 'SMSP', 'VGPO', 'VMT']

# 定义部分混淆矩阵数据
partial_confMatrix = np.array([[2200   , 0  ,  0  ,  0   , 0],
 [   0, 2200,    0 ,   0 ,   0],
 [  12  ,  0, 2188  ,  0   , 0],
 [   0 ,   6 ,   0 ,2194  ,  0],
 [   0  ,  0  ,  0  ,  0 ,2200]])

# 部分数据对应的标签  元抽样

# partial_labels = ['AM', 'FM', 'MNJ', 'SMSP', 'VMT']
# partial_labels = ['ISRJ', 'R_VGPO', 'RGPO', 'RMT', 'VMT']
partial_labels= ['COMB', 'ISRJ', 'MNJ', 'VMT', 'RGPO']



# 创建一个全零的矩阵，大小为 len(allLabels) x len(allLabels)
num_labels = len(allLabels)
full_confMatrix = np.zeros((num_labels, num_labels), dtype=int)

# 填充部分混淆矩阵数据到完整的混淆矩阵中
for i, row_label in enumerate(partial_labels):
    row_index = allLabels.index(row_label)
    for j, col_label in enumerate(partial_labels):
        col_index = allLabels.index(col_label)
        full_confMatrix[row_index, col_index] = partial_confMatrix[i, j]

# 调整图形大小
fig, ax = plt.subplots(figsize=(10, 8))  # 增加图形大小

# 使用 ConfusionMatrixDisplay 绘制混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=full_confMatrix, display_labels=allLabels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)  # 旋转 x 轴标签

# 添加标题和标签
plt.title('混淆矩阵', fontsize=16)
plt.xlabel('预测类别标签', fontsize=16)
plt.ylabel('真实类别标签', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图形
plt.show()
