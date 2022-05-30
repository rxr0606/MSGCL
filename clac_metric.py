import numpy as np

'''
    #真实值：原始邻接47505 一维数组 预测值：47505 模型跑出来的
'''
def get_metrics(real_score, predict_score): #real_score 478023
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))#set函数：去掉重复的元素；展开并由小到大排序得到477218的一维数组
    sorted_predict_score_num = len(sorted_predict_score) #477218
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)] #对42822*[1,2...999]/1000 以一定的间隔取出sorted_predict_score的值作为阈值 大小：999的一维数组
    thresholds = np.mat(thresholds) #将数组转变为1*999的矩阵
    thresholds_num = thresholds.shape[1] #999

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1)) #np.tile：对predict_score沿Y轴复制999次，沿X轴复制1次 999*47505
    negative_index = np.where(predict_score_matrix < thresholds.T) #返回值小于阈值的元素坐标（[行坐标],[列坐标]）
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0 #将对应的坐标值赋值为0
    predict_score_matrix[positive_index] = 1 #999*47505 至此变成了全为0/1的矩阵
    TP = predict_score_matrix.dot(real_score.T) #999*47505点乘47505*1 长度为999的数组[494,494...]
    FP = predict_score_matrix.sum(axis=1)-TP #长度999
    FN = real_score.sum()-TP #999
    FN1=FN
    TN = len(real_score.T)-TP-FP-FN1.numpy() #999

    fpr = FP/(FP+TN) #999
    tpr = TP/(TP+FN.numpy())
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T #2*999
    ROC_dot_matrix.T[0] = [0, 0] #2*999
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]] #2*1000
    x_ROC = ROC_dot_matrix[0].T #1000*1
    y_ROC = ROC_dot_matrix[1].T #1000*1
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:]) #1*1

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN) #999
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list) #989
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]
