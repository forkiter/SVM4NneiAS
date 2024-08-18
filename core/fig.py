# -*- coding = utf-8 -*-
# Plot.
# 2024.07.08,edit by Lin

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from core.train_data import get_data, get_eigenvector_all
import matplotlib

matplotlib.use('TkAgg')

from pylab import mpl
# mpl.rcParams["font.sans-serif"] = ["SimSun"]
# mpl.rcParams["axes.unicode_minus"] = False


def mean_spe_fig(e_types, data_path, gui=False, save_name='mean_spec.png'):
    mean_spe = {}
    for e_type in e_types:
        print('正在计算{}的平均速度谱...'.format(e_type))
        e_path = os.path.join(data_path, e_type)
        e_type_f_list = []
        e_type_spc_list = []
        for root, dirs, files in os.walk(e_path):
            f_list = []
            spc_list = []
            if not files:
                continue
            for file in files:
                spe = get_eigenvector_all(os.path.join(root, file), [], [], mean_fre_fig=True)
                f = spe[0]
                spc = spe[1]
                f_list.append(f)
                spc_list.append(spc)
            f_event = np.mean(np.array(f_list), axis=0)
            spc_event = np.mean(np.array(spc_list), axis=0)
            e_type_f_list.append(f_event)
            e_type_spc_list.append(spc_event)
        f_e_type = np.mean(np.array(e_type_f_list), axis=0)
        spc_e_type = np.mean(np.array(e_type_spc_list), axis=0)
        mean_spe[e_type] = {}
        mean_spe[e_type]['f'] = f_e_type
        mean_spe[e_type]['spc'] = spc_e_type

    print('开始绘图...')
    fig = plt.figure(figsize=(5.2, 3.8), dpi=100)
    # color_list = ["#67D5B5", "#EE7785", "#C89EC4", "#84B1ED", "#f9c00c"]
    color_list = ["#B22222", "#228B22", "#FFD700", "#4876FF", "#B452CD"]
    count = 0
    for key, value in mean_spe.items():
        x = value['f'][:3000]
        y = value['spc'][:3000] / np.max(value['spc'][:3000])
        plt.plot(x, y, color=color_list[count], label=key)
        count += 1
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Velocity amplitude spectrum')

    plt.savefig(os.path.join('out', 'img', save_name), dpi=300)
    if not gui:
        plt.show()
    print('绘图完成.')
    return fig


def opt_fig(opt_values, gui=False, save_name='opt_value.png'):
    print('开始绘图...')
    fig = plt.figure(figsize=(5.2, 3.8), dpi=100)
    x = []
    y = []
    scores = [row[1] for row in opt_values]
    opt_num = scores.index(max(scores))
    opt_score = float(opt_values[opt_num][1])
    error_value = []
    for opt_value in opt_values:
        x.append(opt_value[0])
        y.append(opt_value[1])
        error_value.append(opt_value[2])
    plt.errorbar(x, y, marker='o', color='k', markeredgecolor='k', markerfacecolor='none', markersize=6,
                 yerr=error_value, elinewidth=2, ecolor='k', capsize=3)
    plt.scatter(opt_num, opt_score, color='red', marker='o')
    plt.grid(linestyle='--', color='lightgray')
    # plt.xlabel('Number')
    plt.xlabel('特征参数数量')
    # plt.ylabel('Score')
    plt.ylabel('F1分数')

    plt.savefig(os.path.join('out', 'img', save_name), dpi=300)
    if not gui:
        plt.show()
    print('绘图完成.')

    return fig


def ps_spec_fig(e_types, save_name, p_type, gui=False):
    print('开始绘图...')
    ps_dict = {}
    for e_type in e_types:
        try:
            path = os.path.join('out', 'eigs', 'all_' + e_type + '.txt')
            data = get_data(path)
        except FileNotFoundError:
            raise FileNotFoundError('未找到all_{}.txt文件'.format(e_type))
        y = np.mean(data, axis=0)
        amp_ps = np.divide(data[:, :37], data[:, 37:74])
        y_ps = np.mean(amp_ps, axis=0)
        error_b = np.std(data, axis=0)
        error_b = error_b / np.max(y)
        error_ps = np.std(amp_ps, axis=0)
        error_ps = error_ps / np.max(y_ps)
        ps_dict[e_type] = {}
        ps_dict[e_type]['y'] = y
        ps_dict[e_type]['y_ps'] = y_ps
        ps_dict[e_type]['error_b'] = error_b
        ps_dict[e_type]['error_ps'] = error_ps

    fc_1 = np.arange(0.2, 1, 0.1)
    fc_2 = np.arange(1, 15.5, 0.5)
    x = np.append(fc_1, fc_2)
    color_list = ["#B22222", "#228B22", "#FFD700", "#4876FF", "#B452CD"]
    count = 0
    fig = plt.figure(figsize=(5.2, 3.8), dpi=100)
    for key, value in ps_dict.items():
        if p_type == 'p':
            y = value['y'][:37]
            error_b = value['error_b'][:37]
            plt.ylabel('Ap')
            # plt.ylabel('P波归一化频谱值')
        elif p_type == 's':
            y = value['y'][37:74]
            error_b = value['error_b'][37:74]
            plt.ylabel('As')
            # plt.ylabel('S波归一化频谱值')
        elif p_type == 'ps':
            y = value['y_ps']
            error_b = value['error_ps']
            plt.ylabel('Ap/As')
        else:
            raise ValueError('请输入正确的p_type参数.')

        # if key == 'earthquake':
        #     legend_label = '天然地震'
        # elif key == 'explosion':
        #     legend_label = '爆破'
        # else:
        #     legend_label = '塌陷'

        plt.errorbar(x, y, marker='.', color=color_list[count], markeredgecolor=color_list[count],
                     markerfacecolor='none', markersize=1, yerr=error_b, elinewidth=2, ecolor=color_list[count],
                     capsize=3, label=key)
        count += 1
    plt.xlabel('Frequency（Hz）')
    # plt.xlabel('中心频率（Hz）')
    plt.legend()
    plt.savefig(os.path.join('out', 'img', save_name), dpi=300)
    if not gui:
        plt.show()
    print('绘图完成.')
    return fig


def confusion_matrix_fig(cm, e_types, save_name, gui=False):
    print('开始绘图...')
    shape = len(e_types)
    proportion = []
    for i in cm:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(shape, shape)
    pshow = np.array(pshow).reshape(shape, shape)

    fig = plt.figure(figsize=(5.2, 3.8), dpi=100)
    config = {"font.family": 'Times New Roman'}
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)

    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(shape)
    plt.xticks(tick_marks, e_types, fontsize=12)
    plt.yticks(tick_marks, e_types, fontsize=12)

    iters = np.reshape([[[i, j] for j in range(shape)] for i in range(shape)], (cm.size, 2))
    for i, j in iters:
        if i == j:
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=12, color='white',
                     weight=5)
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='white')
        else:
            plt.text(j, i - 0.12, format(cm[i, j]), va='center', ha='center', fontsize=12)
            plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)

    plt.ylabel('True label', fontsize=12, fontweight='bold')
    plt.xlabel('Predict label', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(os.path.join('out', 'img', save_name), dpi=300)
    if not gui:
        plt.show()
    print('绘图完成.')
    return fig
