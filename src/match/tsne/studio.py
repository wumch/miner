#!/data/pyenv/keras/bin/python

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from pandas.plotting import radviz


class MatBuilder:

    def __init__(self):
        self.col_map = {}
        self.row_map = {}
        self.map = {}

    def feed(self, row_name, col_name):
        col_id = self.col_map.get(col_name, None)
        if col_id is None:
            col_id = self.col_map[col_name] = len(self.col_map)
        row_id = self.row_map.get(row_name, None)
        if row_id is None:
            row_id = self.row_map[row_name] = len(self.row_map)
            self.map[row_id] = []
        self.map[row_id].append(col_id)

    def mat(self) -> np.ndarray:
        arr = np.zeros((len(self.row_map), len(self.col_map)), dtype=np.bool)
        for row_id, col_list in self.map.items():
            for col_id in col_list:
                arr[row_id, col_id] = np.True_
        return arr


mat_builder = MatBuilder()
input_file = '/data/code/comment/doc/dataset/user-task.csv'
for user_id, task_id in (map(int, line.split(',')) for line in open(input_file)):
    mat_builder.feed(row_name=user_id, col_name=task_id)
mat = mat_builder.mat()
X = mat[:, :]
print('matrix shape: {}'.format(X.shape))

fig = plt.figure(figsize=(8, 8))
plt.suptitle("user-task scatter vision in 2-dim. (%d * %d)" % X.shape, fontsize=14)

'''t-SNE'''
t0 = time()
tsne = manifold.TSNE(
    n_components=2, perplexity=20, learning_rate=20.0,
    init='pca', metric='hamming', method='barnes_hut',
    random_state=2)
Y = tsne.fit_transform(X)  # 转换后的输出
t1 = time()
print("t-SNE: %.2f sec" % (t1 - t0))  # 算法用时
ax = fig.add_subplot(1, 1, 1)
plt.scatter(Y[:, 0], Y[:, 1])
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')

plt.show()
