# -*- coding = utf-8 -*-
# Get the eigenvector from original data
# 2024.07.01,edit by Lin

import os
import math
import numpy as np
from obspy import read
from scipy import signal
from scipy.fftpack import fft, ifft


def get_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        file_lines = file.readlines()
        data_list = []
        for file_line in file_lines:
            file_line = file_line.strip("\n").split()
            file_line = [float(i) for i in file_line]
            data_list.append(file_line)
    data_list = np.array(data_list)

    return data_list


def write_data(path, data):
    with open(path, 'w', encoding='utf-8') as file1:
        for line in data:
            fea = ''
            for f in line:
                fea += str(f) + ' '
            fea += '\n'
            file1.write(fea)


def round_up(number, power=0):
    digit = 10 ** power
    num2 = float(int(number * digit))
    if number >= 0 and power != 0:
        tag = number * digit - num2 + 1 / (digit * 10)
        if tag >= 0.5:
            return (num2 + 1) / digit
        else:
            return num2 / digit
    elif number >= 0 and power == 0:
        tag = number * digit - int(number)
        if tag >= 0.5:
            return (num2 + 1) / digit
        else:
            return num2 / digit
    elif power == 0 and number < 0:
        tag = number * digit - int(number)
        if tag <= -0.5:
            return (num2 - 1) / digit
        else:
            return num2 / digit
    else:
        tag = number * digit - num2 - 1 / (digit * 10)
        if tag <= -0.5:
            return (num2 - 1) / digit
        else:
            return num2 / digit


def normalize_wave(data_fft, rate, nor=True):
    data = signal.detrend(data_fft)
    y = fft(data)
    length = np.size(data)
    p2 = np.abs(y / length)
    p1 = p2[:math.floor(length / 2) + 1]
    p1[1:-2] = 2 * p1[1:-2]
    if nor:
        p1 = p1 / np.max(p1)
    f = rate * np.arange(0, length / 2) / length
    return f, p1


def fft_band(tr, fre_min, fre_max):
    dt = tr.stats.delta
    n = len(tr.data)
    y = fft(tr.data)
    yy = []
    for m in range(0, n):
        if fre_min < m / (n * dt) < fre_max or (1 / dt - fre_max) < m / (n * dt) < (1 / dt - fre_min):
            yy.append(y[m])
        else:
            yy.append(0)
    yy = np.array(yy)

    data_fft = np.real(ifft(yy))

    return data_fft


def fc_ei(f, spc, fcs):
    spe_f = []
    for fc in fcs:
        fc = round_up(fc, 1)
        if fc > 1:
            idx = np.where((f > fc - 0.5) & (f <= fc + 0.5))
        else:
            idx = np.where((f > fc - 0.1) & (f <= fc + 0.1))
        afi = spc[idx]
        if len(afi) != 0:
            afi_mean = np.mean(afi)
            spe_f.append(afi_mean)
        else:
            spe_f.append(0)
    return spe_f


def amp_ratio(data_p, data_s, f_p, f_s, fre_min, fre_max):
    idx_p = np.where((f_p >= fre_min) & (f_p <= fre_max))
    idx_s = np.where((f_s >= fre_min) & (f_s <= fre_max))
    try:
        max_amp_p = np.max(data_p[idx_p])
        max_amp_s = np.max(data_s[idx_s])
        ratio = max_amp_p / max_amp_s
    except ValueError:
        ratio = 0

    return ratio


def get_eigenvector_all(sac_file, fcs_p, fcs_s, mean_fre_fig=False):
    stream = read(sac_file)
    st = stream[0]
    head = st.stats
    st = st.detrend('linear')
    fft_data = fft_band(st, fre_min=0.1, fre_max=15)

    p_arrive = int(head.sac.t1 / head.delta)
    p_end = int(head.sac.t3 / head.delta)
    s_arrive = int(head.sac.t2 / head.delta)
    s_end = int(head.sac.t4 / head.delta)

    p_data = fft_data[p_arrive:p_end + 1]
    s_data = fft_data[s_arrive:s_end + 1]
    f_p, spc_p = normalize_wave(p_data, head.sampling_rate)
    f_s, spc_s = normalize_wave(s_data, head.sampling_rate)
    spe_fp = fc_ei(f_p, spc_p, fcs_p)
    spe_fs = fc_ei(f_s, spc_s, fcs_s)
    spe = spe_fp + spe_fs

    f_p, spc_p = normalize_wave(p_data, head.sampling_rate, nor=False)
    f_s, spc_s = normalize_wave(s_data, head.sampling_rate, nor=False)

    spe.append(amp_ratio(spc_p, spc_s, f_p, f_s, 0.1, 1))
    spe.append(amp_ratio(spc_p, spc_s, f_p, f_s, 1, 5))
    spe.append(amp_ratio(spc_p, spc_s, f_p, f_s, 5, 10))
    spe.append(amp_ratio(spc_p, spc_s, f_p, f_s, 10, 15))
    spe.append(amp_ratio(spc_p, spc_s, f_p, f_s, 1, 15))

    if mean_fre_fig:
        f_ps, spc_ps = normalize_wave(fft_data, head.sampling_rate)
        spe = [f_ps, spc_ps]

    return spe


def get_train_data_all(in_dir, out_file, fcs_p, fcs_s):
    feature_list = []

    for root, dirs, files in os.walk(in_dir):
        spe_list = []
        if not files:
            continue
        for file in files:
            spe = get_eigenvector_all(os.path.join(root, file), fcs_p, fcs_s)
            spe_list.append(spe)
        feature = np.array(spe_list)
        feature = np.mean(feature, axis=0)
        feature_list.append(feature)

    if os.path.exists(out_file):
        os.remove(out_file)
    write_data(out_file, feature_list)


def get_std_idx(all_files):
    data = []
    for i, all_file in enumerate(all_files):
        data_np = get_data(all_file)
        data.append(np.mean(data_np, 0))
    data = np.array(data)
    p_data_mean = data[:, 0:37]
    s_data_mean = data[:, 37:74]
    p_std = np.std(p_data_mean, 0)
    p_idx = np.argsort(p_std)[::-1]
    s_std = np.std(s_data_mean, 0)
    s_idx = np.argsort(s_std)[::-1]
    idx = np.hstack((p_idx, s_idx + 37, np.arange(74, np.shape(data)[1], 1)))

    fc_1 = np.arange(0.2, 1, 0.1)
    fc_2 = np.arange(1, 15.5, 0.5)
    fcs = np.append(fc_1, fc_2)
    fcs = fcs.tolist()
    p_fcs = []
    s_fcs = []
    for x in p_idx:
        p_fcs.append(round_up(fcs[x], 1))
    for x in s_idx:
        s_fcs.append(round_up(fcs[x], 1))

    return idx, p_fcs, s_fcs
