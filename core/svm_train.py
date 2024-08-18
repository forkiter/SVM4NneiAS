# -*- coding = utf-8 -*-
# SVM train
# 2024.07.05,edit by Lin

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from core.train_data import get_data


def get_train_data(files, idx, size=0.3, opt_rs=3):
    data = []
    label_data = []
    for i, file in enumerate(files):
        data_np = get_data(file)
        data_np = data_np[:, idx]
        data.append(data_np)
        label_data.append(np.ones(np.shape(data_np)[0]) * i)
    data = np.vstack(data)
    label_data = np.hstack(label_data)

    x_train, x_test, y_train, y_test = train_test_split(data, label_data, test_size=size, random_state=opt_rs)

    return x_train, x_test, y_train, y_test


def svm_pre(x_data, y_data, c=10):
    pre_model = svm.SVC(C=c, kernel='rbf', degree=3, gamma='auto',
                        coef0=0.0, shrinking=True, probability=True, tol=0.0001,
                        cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                        decision_function_shape='ovo', random_state=True)
    pre_model.fit(x_data, y_data)

    return pre_model
