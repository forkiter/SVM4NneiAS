# -*- coding = utf-8 -*-
# Main class for the program running.
# 2024.07.05,edit by Lin

import os
import json
import shutil
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import sklearn.metrics as sm
from core.data_process import check_data, sta_info, get_phases, get_sacs
from core.train_data import get_train_data_all, get_data, write_data
from core.find_opt import find_opt_eigs, get_opt_idx, json_dump
from core.svm_train import get_train_data, svm_pre
from core.svm_predict import s_predict
from core.fig import mean_spe_fig, opt_fig, ps_spec_fig, confusion_matrix_fig


class NneIa:
    def __init__(self):
        path = [os.path.join('data', 'earthquake'), os.path.join('data', 'explosion'),
                os.path.join('data', 'mining'), os.path.join('data', 'reservoir'),
                os.path.join('data', 'subsidence')]
        e_dict = check_data(path)

        e_type_list = []
        e_type_cor = {}
        count = 0
        for e_type, values in e_dict.items():
            path = os.path.abspath(os.path.join('data', e_type))
            e_type_list.append([e_type, path, values['number']])
            e_type_cor[count] = e_type
            count += 1

        self.data = e_dict
        self.e_type_cor = e_type_cor
        self.show_data = e_type_list
        self.station = sta_info(os.path.join('data', 'stations.dat'))
        self.out_data_path = os.path.join('out', 'sac_data')
        if not os.path.exists(self.out_data_path):
            os.mkdir(self.out_data_path)
        self.out_eigs_path = os.path.join('out', 'eigs')
        if not os.path.exists(self.out_eigs_path):
            os.mkdir(self.out_eigs_path)
        if not os.path.exists(os.path.join('out', 'img')):
            os.mkdir(os.path.join('out', 'img'))
        self.opt_value_path = os.path.join('out', 'eigs', 'opt_value.txt')
        self.eigs_config_path = os.path.join('out', 'eigs_config.json')
        self.model_config_path = os.path.join('out', 'model_config.json')
        self.out_model_file_path = os.path.join('out', 'svm_model.pkl')
        self.cf_report_path = os.path.join('out', 'classification_report.json')

        if not os.path.exists(os.path.join('data', 'config.json')):
            raise FileNotFoundError('未找到config.json文件。')
        with open(os.path.join('data', 'config.json'), 'r') as jsonfile:
            config = json.load(jsonfile)
        self.config = config

        self.opt_num = None
        self.opt_value = None
        self.all_eigs_list = None
        self.opt_idx = None
        self.opt_p_fc = None
        self.opt_s_fc = None
        self.opt_rs = None
        self.cf_report = None
        self.confusion_matrix = None

    def data2sac(self):
        print('截取数据段配置如下：')
        for key, values in self.config.items():
            dis = key.split('-')
            if values[0] == -1:
                p_value = ' S-P '
            else:
                p_value = ' ' + str(values[0]) + ' '
            print("{}km～{}km：P波后{}秒，S波后{}秒。".format(dis[0], dis[1], p_value, ' ' + str(values[1]) + ' '))
        for e_type in self.data.keys():
            out_path = os.path.join(self.out_data_path, e_type)
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            if os.listdir(out_path):
                for file in Path(out_path).glob('*'):
                    file.unlink() if file.is_file() else shutil.rmtree(file)

            print('正在处理{}数据...'.format(e_type))

            ps_arrive = get_phases(os.path.join('data', e_type), self.station, self.config)
            print(ps_arrive)
            get_sacs(os.path.join('data', e_type), out_path, ps_arrive, sac_len=199.99)
        print('数据处理完成。')

    def get_all_eigs(self):
        eigs_file_list = []
        fc_1 = np.arange(0.2, 1, 0.1)
        fc_2 = np.arange(1, 15.5, 0.5)
        fcs = np.append(fc_1, fc_2)
        for e_type in self.data.keys():
            out_path = os.path.join(self.out_data_path, e_type)
            if not os.path.exists(self.out_eigs_path):
                os.mkdir(self.out_eigs_path)
            all_eigs_file = 'all_' + e_type + '.txt'
            all_eigs_file_path = os.path.join(self.out_eigs_path, all_eigs_file)
            if os.path.exists(all_eigs_file_path):
                os.remove(all_eigs_file_path)

            print('正在计算{}所有的特征值...'.format(e_type))
            get_train_data_all(out_path, all_eigs_file_path, fcs, fcs)
            eigs_file_list.append(all_eigs_file_path)
        self.all_eigs_list = eigs_file_list
        json_dump(self.all_eigs_list, self.eigs_config_path)
        print('所有特征值已计算完毕。')

    def get_opt_eigs(self):
        if self.all_eigs_list is None:
            all_eigs_list = []
            for file in os.listdir(self.out_eigs_path):
                if file.startswith('all_'):
                    all_eigs_list.append(os.path.join(self.out_eigs_path, file))
            self.all_eigs_list = all_eigs_list
        print('开始计算最佳特征值数据...')
        opt_value = find_opt_eigs(self.all_eigs_list, 38)
        scores = [row[1] for row in opt_value]
        self.opt_num = scores.index(max(scores))
        self.opt_value = opt_value

        if os.path.exists(self.opt_value_path):
            os.remove(self.opt_value_path)
        write_data(self.opt_value_path, opt_value)
        print('特征值曲线已生成。')

    def get_svm_model(self, opt_num=None):
        if self.all_eigs_list is None:
            all_eigs_list = []
            for file in os.listdir(self.out_eigs_path):
                if file.startswith('all_'):
                    all_eigs_list.append(os.path.join(self.out_eigs_path, file))
            self.all_eigs_list = all_eigs_list
        if self.opt_value is None:
            self.opt_value = get_data(self.opt_value_path)
        if opt_num is None:
            if self.opt_num is None:
                scores = [row[1] for row in self.opt_value]
                self.opt_num = scores.index(max(scores))
        else:
            self.opt_num = opt_num
        print('开始生成模型...')
        if not os.path.exists(self.eigs_config_path):
            json_dump(self.all_eigs_list, self.eigs_config_path)
        eigs_config = json.loads(open(self.eigs_config_path).read())

        opt_rs = int(self.opt_value[self.opt_num][3])
        self.opt_rs = opt_rs
        out_idx, out_p_fc, out_s_fc = get_opt_idx(eigs_config['all_idx'], eigs_config['p_fcs'], eigs_config['s_fcs'],
                                                  opt_num=self.opt_num)
        self.opt_idx = out_idx
        self.opt_p_fc = out_p_fc
        self.opt_s_fc = out_s_fc
        x_train, x_test, y_train, y_test = get_train_data(self.all_eigs_list, idx=self.opt_idx, opt_rs=self.opt_rs)
        svm_model = svm_pre(x_train, y_train)
        if os.path.exists(self.out_model_file_path):
            os.remove(self.out_model_file_path)
        joblib.dump(svm_model, self.out_model_file_path)

        data = {
            'svm_model': self.out_model_file_path,
            'opt_p_fc': self.opt_p_fc,
            'opt_s_fc': self.opt_s_fc,
            'opt_num': self.opt_num,
            'e_type_cor': self.e_type_cor
        }
        with open(self.model_config_path, 'w') as jsonfile:
            json.dump(data, jsonfile)

        y_pre = svm_model.predict(x_test)
        self.cf_report = classification_report(y_test, y_pre, output_dict=True)
        self.confusion_matrix = sm.confusion_matrix(y_test, y_pre)

        with open(self.cf_report_path, 'w') as jsonfile:
            json.dump(self.cf_report, jsonfile)
        print('模型已生成。')

    def mean_fre_fig(self, gui, save_name):
        fig = mean_spe_fig(self.data.keys(), self.out_data_path, gui=gui, save_name=save_name)
        return fig

    def opt_fig(self, gui, save_name):
        if self.opt_value is None:
            self.opt_value = get_data(self.opt_value_path)
        fig = opt_fig(self.opt_value, gui=gui, save_name=save_name)

        return fig

    def ps_fre_fig(self, save_name, p_type, gui):
        fig = ps_spec_fig(self.data.keys(), save_name=save_name, p_type=p_type, gui=gui)
        return fig

    def cm_fig(self, save_name, gui):
        if self.all_eigs_list is None:
            all_eigs_list = []
            for file in os.listdir(self.out_eigs_path):
                if file.startswith('all_'):
                    all_eigs_list.append(os.path.join(self.out_eigs_path, file))
            self.all_eigs_list = all_eigs_list
        if self.opt_num is None:
            model_config = json.loads(open(self.model_config_path).read())
            self.opt_num = int(model_config['opt_num'])
        if self.opt_idx is None:
            eigs_config = json.loads(open(self.eigs_config_path).read())

            self.opt_idx, _, _ = get_opt_idx(eigs_config['all_idx'], eigs_config['p_fcs'],
                                             eigs_config['s_fcs'], opt_num=self.opt_num)
        if self.opt_rs is None:
            self.opt_value = get_data(self.opt_value_path)
            opt_rs = int(self.opt_value[self.opt_num][3])
            self.opt_rs = opt_rs
        x_train, x_test, y_train, y_test = get_train_data(self.all_eigs_list, idx=self.opt_idx, opt_rs=self.opt_rs)
        svm_model = joblib.load(self.out_model_file_path)
        y_pre = svm_model.predict(x_test)
        self.confusion_matrix = sm.confusion_matrix(y_test, y_pre)
        fig = confusion_matrix_fig(self.confusion_matrix, self.data.keys(), save_name=save_name, gui=gui)
        return fig


class NPredict:
    def __init__(self):
        self.input_path = 'predict_data'
        self.out_path = os.path.join(self.input_path, 'temp')
        self.out_result = os.path.join(self.input_path, 'predict_result.txt')
        self.station = sta_info(os.path.join('data', 'stations.dat'))
        self.config = json.loads(open(os.path.join('data', 'config.json')).read())
        self.model = joblib.load(os.path.join('out', 'svm_model.pkl'))
        self.model_config = json.loads(open(os.path.join('out', 'model_config.json')).read())
        self.p_fc = self.model_config['opt_p_fc']
        self.s_fc = self.model_config['opt_s_fc']
        self.e_type_cor = self.model_config['e_type_cor']

        self.pre_data = {}
        count = 1
        for file in os.listdir(self.input_path):
            if file.endswith('.seed'):
                file_name = file[:-5]
                file_path = os.path.join(self.input_path, file)
                file_phase = os.path.join(self.input_path, file_name + '.phase')
                if not os.path.exists(file_phase):
                    raise FileNotFoundError('{}震相文件未找到'.format(file_phase))
                self.pre_data[count] = [file_name, file_path]
                count += 1

    def pre(self):
        if os.path.exists(self.out_path):
            shutil.rmtree(self.out_path)
        os.mkdir(self.out_path)
        out_eigs_path = os.path.join(self.out_path, 'eigs.txt')
        pre_e_type, pre_prob_list = s_predict(self.input_path, self.out_path, out_eigs_path, self.station, self.model,
                                              self.p_fc, self.s_fc, self.config, self.e_type_cor)
        shutil.rmtree(self.out_path)

        wd_list = [['ID', '文件名', '预测结果']]
        for value in self.e_type_cor.values():
            wd_list[0].append(value + '概率')
        for i in range(1, len(self.pre_data) + 1):
            w_id = i
            w_file = self.pre_data[i][0]
            w_pre_result = pre_e_type[i - 1]
            wd_list.append([w_id, w_file, w_pre_result, pre_prob_list[i - 1][0][0], pre_prob_list[i - 1][0][1],
                            pre_prob_list[i - 1][0][2]])
        write_data(self.out_result, wd_list)
        print('预测结果已生成。')

        return pre_e_type, pre_prob_list
