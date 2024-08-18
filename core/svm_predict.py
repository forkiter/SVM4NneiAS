# -*- coding = utf-8 -*-
# Predict
# 2024.07.24,edit by Lin

import numpy as np
from core.data_process import get_phases, get_sacs
from core.train_data import get_train_data_all, get_data


def s_predict(data_path, out_path, out_eigs_file, stations, model, p_fc, s_fc, config, cor):
    ps_arrive = get_phases(data_path, stations, config)
    get_sacs(data_path, out_path, ps_arrive, sac_len=199.99)
    get_train_data_all(out_path, out_eigs_file, p_fc, s_fc)
    data = get_data(out_eigs_file)
    pre_e_type = []
    pre_prob_list = []
    for event in data:
        event = np.reshape(event, (1, len(event)))
        y_pre = model.predict(event)
        e_type = cor[str(int(y_pre[0]))]
        y_prob = model.predict_proba(event)
        pre_e_type.append(e_type)
        pre_prob_list.append(y_prob)

    return pre_e_type, pre_prob_list
