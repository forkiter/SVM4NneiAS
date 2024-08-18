# -*- coding = utf-8 -*-
# EXAMPLE
# 具体操作请详见"/docs/UserGuide.pdf"
# 2024.07.24,edit by Lin

from core.nneias import NneIa, NPredict

''' 创建实例 '''
nne = NneIa()
print(nne.show_data)  # 查看实例是否创建成功，查看载入数据属性


''' 预处理 '''
nne.data2sac()  # 数据预处理


''' 计算特征值 '''
nne.get_all_eigs()
nne.get_opt_eigs()


''' 生成模型 '''
nne.get_svm_model()


''' 预测 '''
npr = NPredict()
npr.pre()


''' 绘制平均速度谱图 '''
nne.mean_fre_fig(gui=False, save_name='mean_spec.png')


''' 绘制特征值曲线图 '''
nne.opt_fig(gui=False, save_name='opt_value.png')


''' 绘制P波、S波、P/S频谱图 '''
nne.ps_fre_fig(save_name='p_spec.png', p_type='p', gui=False)
nne.ps_fre_fig(save_name='s_spec.png', p_type='s', gui=False)
nne.ps_fre_fig(save_name='ps_spec.png', p_type='ps', gui=False)


''' 绘制SVM混淆矩阵图 '''
nne.cm_fig(save_name='confusion_matrix.png', gui=False)

