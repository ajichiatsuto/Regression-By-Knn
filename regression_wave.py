import mglearn as mg
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

X, y = mg.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
# ここでのスコアはR^2スコアと呼ばれる決定係数と呼ばれるもので、回帰モデルの性能を評価する指標
# 1は完璧な予測、0は常にy_trainの平均値を予測するモデル
print("test set accuracy: {:.2f}".format(reg.score(X_test, y_test)))

