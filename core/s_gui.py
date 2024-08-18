# -*- coding = utf-8 -*-
# GUI for SVM4NneiAS
# 2024.07.22,edit by Lin

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter.messagebox import askyesno
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from core.train_data import round_up
from core.nneias import NneIa, NPredict


class NiGui(object):
    def __init__(self, init_window_name):
        self.nei = None
        self.pre_n = None
        self.file_input_dirs = None
        self.canvas = None
        self.init_window_name = init_window_name
        self.init_window_name.title('SVM4NneiAS v1.0')
        self.init_window_name.geometry('800x600')
        self.init_window_name.iconbitmap(os.path.join('docs', 'fig', 'head.ico'))

        self.init_window_name.protocol('WM_DELETE_WINDOW', self.close_window)

        """ 组件容器 """
        self.left_frame = tk.Frame(self.init_window_name)
        self.left_frame.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5)
        self.data_process_frame = tk.LabelFrame(self.left_frame, text='重新训练', padx=5, pady=10)
        self.data_process_frame.pack()
        self.model_frame = tk.LabelFrame(self.left_frame, text='生成模型', padx=5, pady=10)
        self.model_frame.pack()
        self.plot_frame = tk.LabelFrame(self.left_frame, text='绘图', padx=5, pady=10)
        self.plot_frame.pack()
        self.pre_frame = tk.LabelFrame(self.left_frame, text='结果预测', padx=5, pady=10)
        self.pre_frame.pack()

        self.right_frame = tk.Frame(self.init_window_name)
        self.right_frame.pack(side=tk.TOP, padx=5, pady=5)
        self.show_data_frame = tk.Frame(self.right_frame)
        self.show_data_frame.pack()
        tk.Label(self.show_data_frame, text="数据显示窗口").pack(anchor=tk.W)
        self.data_pad = tk.Text(self.show_data_frame, width=80, height=30)
        self.data_pad.config(state='normal')
        self.data_pad.pack(side=tk.LEFT, fill=tk.X)

        """ 数据显示窗和日志显示窗 """
        self.listbox = ttk.Treeview(self.data_pad, show='headings', height=18)
        self.data_bar = ttk.Scrollbar(self.show_data_frame, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.data_bar.set)
        heading = ['ID', '类型', '位置', '数量']
        self.listbox.configure(columns=heading)
        for col in heading:
            if col == 'ID':
                self.listbox.column(col, width=40, anchor='center')
                self.listbox.heading(col, text=col)
            elif col == '位置':
                self.listbox.column(col, width=300, anchor='center')
                self.listbox.heading(col, text=col)
            elif col == '数量':
                self.listbox.column(col, width=80, anchor='center')
                self.listbox.heading(col, text=col)
            else:
                self.listbox.column(col, width=120, anchor='center')
                self.listbox.heading(col, text=col)
        self.listbox.pack()
        self.data_bar.pack(side=tk.RIGHT, fill=tk.Y)

        self.show_log_frame = tk.Frame(self.right_frame)
        self.show_log_frame.pack()
        tk.Label(self.show_log_frame, text="日志显示窗口").pack(anchor=tk.W)
        self.log_pad = ScrolledText(self.show_log_frame, width=80, height=11)
        self.log_pad.pack(side=tk.LEFT, fill=tk.X)
        self.log_pad.tag_configure('stderr', foreground='#b22222')

        self.file_confirm_button = tk.Button(self.data_process_frame, text="数据载入", command=lambda: self.load_data())
        self.file_confirm_button.grid(padx=10, pady=8, row=0, column=0)
        self.file_confirm_button = tk.Button(self.data_process_frame, text="数据预处理", command=lambda: self.pro_data())
        self.file_confirm_button.grid(padx=10, pady=8, row=0, column=1)
        self.file_confirm_button = tk.Button(self.data_process_frame, text="计算所有特征值", command=lambda: self.all_eigs())
        self.file_confirm_button.grid(padx=10, pady=8, row=1, column=0)
        self.file_confirm_button = tk.Button(self.data_process_frame, text="计算最佳特征值", command=lambda: self.opt_eigs())
        self.file_confirm_button.grid(padx=10, pady=8, row=1, column=1)

        self.k_input_title = tk.Label(self.model_frame, text="特征值数量 =")
        self.k_input_title.grid(padx=10, pady=8, row=0, column=0, sticky=tk.W)
        self.k_input_entry = tk.Entry(self.model_frame, width=17)
        self.k_input_entry.grid(padx=3, pady=8, row=0, column=1, sticky=tk.E)
        self.model_confirm_button = tk.Button(self.model_frame, text="开始生成", command=lambda: self.c_model())
        self.model_confirm_button.grid(padx=10, pady=8, row=1, column=0, sticky=tk.W)

        """ 画图组件 """
        self.plot_choose_type = ttk.Combobox(self.plot_frame, width=29)
        self.plot_choose_type['values'] = ['平均速度谱图', '特征值曲线图', 'P波频谱图', 'S波频谱图',
                                           'P/S谱振幅比图', 'SVM测试混淆矩阵图']
        self.plot_choose_type.current(0)
        self.plot_choose_type.grid(padx=0, pady=3, row=1, column=0, sticky=tk.W)
        self.plot_save_name_title = tk.Label(self.plot_frame, text="图片名：")
        self.plot_save_name_title.grid(padx=0, pady=3, row=3, column=0, sticky=tk.W)
        self.plot_save_name_entry = tk.Entry(self.plot_frame, width=17)
        self.plot_save_name_entry.insert(0, '')
        self.plot_save_name_entry.grid(padx=10, pady=3, row=3, column=0, sticky=tk.E)
        self.plot_confirm_button = tk.Button(self.plot_frame, text="开始绘图", command=lambda: self.plot_run())
        self.plot_confirm_button.grid(padx=20, pady=3, row=4, column=0, sticky=tk.W)
        self.plot_destroy_button = tk.Button(self.plot_frame, text="清除绘图", command=lambda: self.plot_des())
        self.plot_destroy_button.grid(padx=30, pady=3, row=4, column=0, sticky=tk.E)

        self.pre_confirm_button = tk.Button(self.pre_frame, text="数据载入", command=lambda: self.pre_load_data())
        self.pre_confirm_button.grid(padx=33, pady=8, row=0, column=0)
        self.pre_confirm_button = tk.Button(self.pre_frame, text="预测", command=lambda: self.predict_run())
        self.pre_confirm_button.grid(padx=33, pady=8, row=0, column=1)

        """ 重定向输出到log """
        sys.stdout = TextRedirector(self.log_pad, 'stdout')
        sys.stderr = TextRedirector(self.log_pad, 'stderr')

        """ 版本信息及联系方式 """
        self.version = tk.Label(self.left_frame, text='SVM4NneiAS made by Lin, Gdsin\nEmail:  forkiter@163.com')
        self.version.pack(side=tk.LEFT, padx=30, pady=20)

    def load_data(self):
        try:
            nei = NneIa()
        except ValueError as e:
            messagebox.showerror('Error:', str(e))
            return False
        self.nei = nei
        self.show_data()

    def show_data(self):
        show_data = self.nei.show_data
        self.listbox.delete(*self.listbox.get_children())
        for i, row in enumerate(show_data, start=0):
            self.listbox.insert("", "end", values=(i, row[0], row[1], row[2]))

        self.data_pad.config(state='disabled')

    def pro_data(self):
        if self.nei is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        self.nei.data2sac()

    def all_eigs(self):
        if self.nei is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        self.nei.get_all_eigs()

    def opt_eigs(self):
        if self.nei is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        self.nei.get_opt_eigs()
        if self.canvas is not None:
            self.plot_des()
        self.opt_eigs_show()

    def opt_eigs_show(self):
        heading = ['数量值', 'F1平均得分', 'F1最高分']
        self.listbox.configure(columns=heading)
        for col in heading:
            if col == '数量值':
                self.listbox.column(col, width=60, anchor='center')
                self.listbox.heading(col, text=col)
            else:
                self.listbox.column(col, width=240, anchor='center')
                self.listbox.heading(col, text=col)

        self.listbox.delete(*self.listbox.get_children())
        show_data = self.nei.opt_value
        for row in show_data:
            self.listbox.insert("", "end", values=(int(row[0]), round_up(row[1], 3), round_up(row[1] + row[2], 3)))
        self.data_pad.config(state='disabled')

    def c_model(self):
        if self.nei is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        if self.k_input_entry.get().isdigit():
            k = int(self.k_input_entry.get())
        else:
            k = None
        self.nei.get_svm_model(k)
        if self.canvas is not None:
            self.plot_des()
        self.cf_show()

    def cf_show(self):
        heading = ['ID', '类型', '精确率', '召回率', 'F1得分', '测试集数量']
        self.listbox.configure(columns=heading)
        for col in heading:
            if col == 'ID':
                self.listbox.column(col, width=30, anchor='center')
                self.listbox.heading(col, text=col)
            else:
                self.listbox.column(col, width=100, anchor='center')
                self.listbox.heading(col, text=col)

        self.listbox.delete(*self.listbox.get_children())
        show_data = self.nei.cf_report
        for key, value in show_data.items():
            if key == 'accuracy':
                self.listbox.insert("", "end", values=('', key, '', '', round_up(value, 3), ''))
            elif key == 'macro avg' or key == 'weighted avg':
                self.listbox.insert("", "end", values=(
                    '', key, round_up(value['precision'], 3), round_up(value['recall'], 3),
                    round_up(value['f1-score'], 3), int(value['support'])))
            else:
                e_id = int(float(key))
                e_type = self.nei.e_type_cor[e_id]
                self.listbox.insert("", "end", values=(
                    e_id, e_type, round_up(value['precision'], 3), round_up(value['recall'], 3),
                    round_up(value['f1-score'], 3), int(value['support'])))
        self.data_pad.config(state='disabled')

    def plot_run(self):
        if self.nei is None:
            messagebox.showwarning(message='请导入数据。', title='警告')
        plot_type = self.plot_choose_type.get()
        if plot_type == '平均速度谱图':
            save_name = 'mean_spec.png' if self.plot_save_name_entry.get() == '' \
                else self.plot_save_name_entry.get()
            fig = self.nei.mean_fre_fig(gui=True, save_name=save_name)
        elif plot_type == '特征值曲线图':
            save_name = 'opt_value.png' if self.plot_save_name_entry.get() == '' \
                else self.plot_save_name_entry.get()
            fig = self.nei.opt_fig(gui=True, save_name=save_name)
        elif plot_type == 'P波频谱图':
            save_name = 'p_spec.png' if self.plot_save_name_entry.get() == '' \
                else self.plot_save_name_entry.get()
            fig = self.nei.ps_fre_fig(save_name=save_name, p_type='p', gui=True)
        elif plot_type == 'S波频谱图':
            save_name = 's_spec.png' if self.plot_save_name_entry.get() == '' \
                else self.plot_save_name_entry.get()
            fig = self.nei.ps_fre_fig(save_name=save_name, p_type='s', gui=True)
        elif plot_type == 'P/S谱振幅比图':
            save_name = 'ps_spec.png' if self.plot_save_name_entry.get() == '' \
                else self.plot_save_name_entry.get()
            fig = self.nei.ps_fre_fig(save_name=save_name, p_type='ps', gui=True)
        elif plot_type == 'SVM测试混淆矩阵图':
            save_name = 'confusion_matrix.png' if self.plot_save_name_entry.get() == '' \
                else self.plot_save_name_entry.get()
            fig = self.nei.cm_fig(save_name, gui=True)
        else:
            raise ValueError('没有这个选项，请重试.')

        canvas = FigureCanvasTkAgg(fig, master=self.init_window_name)
        canvas.get_tk_widget().place(x=275, y=32)
        canvas.draw()
        self.canvas = canvas

    def pre_load_data(self):
        try:
            pre_n = NPredict()
        except ValueError as e:
            messagebox.showerror('Error:', str(e))
            return False
        self.pre_n = pre_n

        heading = ['ID', '文件名', '文件位置']
        self.listbox.configure(columns=heading)
        for col in heading:
            if col == 'ID':
                self.listbox.column(col, width=30, anchor='center')
                self.listbox.heading(col, text=col)
            elif col == '文件名':
                self.listbox.column(col, width=170, anchor='center')
                self.listbox.heading(col, text=col)
            else:
                self.listbox.column(col, width=300, anchor='center')
                self.listbox.heading(col, text=col)
        self.listbox.delete(*self.listbox.get_children())

        for key, value in self.pre_n.pre_data.items():
            self.listbox.insert("", "end", values=(key, value[0], value[1]))

        self.data_pad.config(state='disabled')
        print('预测数据已载入')

    def predict_run(self):
        if self.pre_n is None:
            messagebox.showwarning(message='请先载入预测数据。', title='警告')
        pre_e_type, pre_prob_list = self.pre_n.pre()

        heading = ['ID', '文件名', '预测结果']
        for value in self.pre_n.e_type_cor.values():
            heading.append(value[:5] + '概率')
        self.listbox.configure(columns=heading)
        for col in heading:
            if col == 'ID':
                self.listbox.column(col, width=20, anchor='center')
                self.listbox.heading(col, text=col)
            elif col == '文件名':
                self.listbox.column(col, width=120, anchor='center')
                self.listbox.heading(col, text=col)
            else:
                self.listbox.column(col, width=80, anchor='center')
                self.listbox.heading(col, text=col)
        self.listbox.delete(*self.listbox.get_children())
        for key, value in self.pre_n.pre_data.items():
            insert_value = [key, value[0], pre_e_type[key - 1]]
            for i in pre_prob_list[key - 1][0]:
                insert_value.append(round_up(i, 3))
            insert_value = tuple(insert_value)
            self.listbox.insert("", "end", values=insert_value)

        self.data_pad.config(state='disabled')

    def plot_des(self):
        self.canvas.get_tk_widget().destroy()

    def run_log_print(self, message):
        """ 实时更新日志，固定用法 """
        self.log_pad.config(state=tk.NORMAL)
        self.log_pad.insert(tk.END, "\n" + message + "\n")
        self.log_pad.see(tk.END)
        self.log_pad.update()
        self.log_pad.config(state=tk.DISABLED)

    def close_window(self):
        """ 退出/关闭窗体 固定方法 """
        ans = askyesno(title='退出警告', message='是否确定退出程序？\n是则退出，否则继续！')
        if ans:
            self.init_window_name.destroy()
            sys.exit()
        else:
            return None


class TextRedirector(object):
    """ 控制台输出重定向只日志显示窗口 """

    def __init__(self, widget, tag='stdout'):
        self.widget = widget
        self.tag = tag

    def write(self, str_):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str_, (self.tag,))
        self.widget.see(tk.END)
        self.widget.update()
        self.widget.configure(state='disabled')
