import matplotlib.pyplot as plt
from sklearn import manifold


def create_embedding(features, **kargs):
    # t-SNEで2次元に圧縮
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    features = tsne.fit_transform(features)
    return features


def visualize(features, labels, num_classes):
    # カラーマップ
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    plt.figure(figsize=(10, 5))
    # 描画
    for i in range(num_classes):
        plt.plot(features[labels == i, 0], features[labels == i, 1], '.', c=colors[i])

    # グラフ設定
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.show()
