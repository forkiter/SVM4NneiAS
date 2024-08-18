# -*- coding = utf-8 -*-
# Find the optimal parameters
# 2024.07.22,edit by Lin

import os
import numpy as np
import json
from core.train_data import get_std_idx
from core.svm_train import get_train_data, svm_pre


def get_opt_idx(idx, p_fc, s_fc, opt_num):
    end_num = np.shape(idx)[0]
    if opt_num == 0:
        out_idx = np.arange(74, end_num, 1)
        out_p_fc = None
        out_s_fc = None
    else:
        out_idx = np.append(idx[:opt_num], idx[37:37 + opt_num])
        out_idx = np.append(out_idx, np.arange(74, end_num, 1))
        out_p_fc = p_fc[:opt_num]
        out_s_fc = s_fc[:opt_num]

    return out_idx, out_p_fc, out_s_fc


def find_opt_eigs(all_files, max_iter=38):
    opt_list = []
    all_idx, p_fcs, s_fcs = get_std_idx(all_files)
    for i in range(max_iter):
        opt_idx, _, _ = get_opt_idx(all_idx, p_fcs, s_fcs, i)
        opt_score_list = []
        for j in range(100):
            x_train, x_test, y_train, y_test = get_train_data(all_files, idx=opt_idx, size=0.1, opt_rs=j)
            svm_model = svm_pre(x_train, y_train)
            score = svm_model.score(x_test, y_test)
            opt_score_list.append(score)
        score_max = max(opt_score_list)
        score_min = min(opt_score_list)
        opt_score = (score_min + score_max) / 2
        error_value = score_max - opt_score
        score_max_idx = opt_score_list.index(score_max)
        opt_list.append([i, opt_score, error_value, score_max_idx])

    return opt_list


def json_dump(files, path):
    all_idx, p_fcs, s_fcs = get_std_idx(files)
    if os.path.exists(path):
        os.remove(path)
    data = {
        'all_idx': all_idx.tolist(),
        'p_fcs': p_fcs,
        's_fcs': s_fcs
    }
    with open(path, 'w') as jsonfile:
        json.dump(data, jsonfile)
