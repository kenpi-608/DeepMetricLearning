import numpy as np
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


def create_classification_dataset(loader, model, device):
    features = []
    labels = []
    for img, label in tqdm(loader):
        img, label = img.to(device), label.to(device)
        pred = model(img)

        features.append(pred.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
    features = np.concatenate(features, 0)
    labels = np.concatenate(labels, 0)
    return features, labels


def plot_confusion_matrix(pred, y):
    plt.figure(figsize=(10, 5))
    # conf_mat = confusion_matrix(y, model.predict(x), normalize='pred')
    sns.heatmap(confusion_matrix(y, pred), annot=True, cmap='Blues', fmt='g')
    plt.show()


def lr(train_x, train_y, test_x, test_y):
    """
    ロジスティック回帰
    """
    clf = LogisticRegression(random_state=0).fit(train_x, train_y)
    print("train acc: ", accuracy_score(train_y, clf.predict(train_x)))
    print("valid acc: ", accuracy_score(test_y, clf.predict(test_x)))
    print("macro mean precision: ", precision_score(test_y, clf.predict(test_x), average='macro'))
    print("macro mean recall: ", recall_score(test_y, clf.predict(test_x), average='macro'))
    plot_confusion_matrix(test_y, clf.predict(test_x))
    return clf


def lightgbm(train_x, train_y, test_x, test_y):
    """
    lightgbm
    """
    train_data = lgb.Dataset(train_x, label=train_y)
    eval_data = lgb.Dataset(test_x, label=test_y, reference=train_data)
    params = {'task': 'train', 'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'num_class': np.unique(train_y).shape[0],
              'verbose': -1}

    gbm = lgb.train(params, train_data, valid_sets=eval_data,
                    num_boost_round=50, verbose_eval=False)
    print("train acc: ", accuracy_score(train_y, create_lgb_pred(gbm, train_x)))
    print("valid acc: ", accuracy_score(test_y, create_lgb_pred(gbm, test_x)))
    print("macro mean precision: ", precision_score(test_y, create_lgb_pred(gbm, test_x), average='macro'))
    print("macro mean recall: ", recall_score(test_y, create_lgb_pred(gbm, test_x), average='macro'))
    plot_confusion_matrix(test_y, create_lgb_pred(gbm, test_x))
    return gbm


def create_lgb_pred(model, x):
    preds = model.predict(x)
    pred_y = []

    for x in preds:
        pred_y.append(np.argmax(x))
    return pred_y
