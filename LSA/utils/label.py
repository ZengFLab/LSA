import re
import numpy as np
import pandas as pd
from numba import njit
from itertools import chain
from joblib import Parallel, delayed
from typing import Literal

class LabelMatrix:
    def __init__(self):
        self.labels_ = None
        self.control_label = None 
        self.sep_pattern = None
    
    def fit_transform(self, labels, control_label=None, sep_pattern=r'[,;_\s]', speedup: Literal['none','vectorize','parallel']='none'):
        if speedup=='none':
            mat, self.labels_ = label_to_matrix(labels=labels, sep_pattern=sep_pattern)
        elif speedup=='vectorize':
            mat, self.labels_ = vectorized_label_to_matrix(labels=labels, sep_pattern=sep_pattern)
        elif speedup=='parallel':
            mat, self.labels_ = parallel_label_to_matrix(labels=labels, sep_pattern=sep_pattern)
            
        self.labels_ = np.array(self.labels_)
        
        if control_label is not None:
            idx = np.where(self.labels_==control_label)[0]
            mat = np.delete(mat, idx, axis=1)
            self.labels_ = np.delete(self.labels_, idx)
            
        self.control_label = control_label
        self.sep_pattern=sep_pattern
        
        return mat
    
    def transform(self, labels, speedup: Literal['none','vectorize','parallel']='none'):
        sep_pattern = self.sep_pattern
        if speedup=='none':
            mat, labels_ = label_to_matrix(labels=labels, sep_pattern=sep_pattern)
        elif speedup=='vectorize':
            mat, labels_ = vectorized_label_to_matrix(labels=labels, sep_pattern=sep_pattern)
        elif speedup=='parallel':
            mat, labels_ = parallel_label_to_matrix(labels=labels, sep_pattern=sep_pattern)
          
        mat_df = pd.DataFrame(mat, columns=labels_)
        
        labels_valid = [x for x in labels_ if x in self.labels_]  
        mat_df = mat_df[labels_valid]
        
        mat_valid = np.zeros([mat.shape[0], len(self.labels_)])
        mat_valid_df = pd.DataFrame(mat_valid, columns=self.labels_)
        mat_valid_df[labels_valid] = mat_df 
        
        return mat_valid_df.values
        
    def inverse_transform(self, matrix):
        return matrix_to_labels(matrix=matrix, unique_labels=self.labels_)
        
def label_to_matrix(labels, sep_pattern=r'[;_\-\s]'):
    """
    将混合分隔符的多标签数据转换为 0-1 矩阵
    
    Args:
        labels: 原始标签列表，如 ["cat", "dog", "cat;dog", "bird_dog"]
        sep_pattern: 多标签分隔符的正则模式（默认匹配 ; _ - 和空格）
    
    Returns:
        one_hot_matrix: 0-1 矩阵
        unique_labels: 唯一标签列表
    """
    # 统一分隔符
    labels_unified = [re.sub(sep_pattern, ';', label) for label in labels]
    
    # 获取所有唯一标签
    all_unique_labels = sorted(set(chain(*[label.split(';') for label in labels_unified])))
    
    # 生成 0-1 矩阵
    matrix = np.zeros((len(labels), len(all_unique_labels)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(all_unique_labels)}
    
    for i, label in enumerate(labels_unified):
        for sub_label in label.split(';'):
            if sub_label in label_to_idx:
                matrix[i, label_to_idx[sub_label]] = 1
    
    return matrix, all_unique_labels


def vectorized_label_to_matrix(labels, sep_pattern=r'[;_\-\s]'):
    labels_unified = [re.sub(sep_pattern, ';', label) for label in labels]
    unique_labels = sorted(set(chain(*[label.split(';') for label in labels_unified])))
    
    # 向量化操作
    label_matrix = np.array([label.split(';') for label in labels_unified], dtype=object)
    matrix = np.zeros((len(labels), len(unique_labels)), dtype=int)
    
    for i, label in enumerate(unique_labels):
        matrix[:, i] = np.array([label in lst for lst in label_matrix], dtype=int)
    
    return matrix, unique_labels



def parallel_label_to_matrix(labels, sep_pattern=r'[;_\-\s]', n_jobs=4):
    labels_unified = [re.sub(sep_pattern, ';', label) for label in labels]
    unique_labels = sorted(set(chain(*[label.split(';') for label in labels_unified])))
    
    def process_row(row_labels, unique_labels):
        return [1 if label in row_labels else 0 for label in unique_labels]
    
    label_lists = [label.split(';') for label in labels_unified]
    matrix = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row, unique_labels) for row in label_lists
    )
    
    return np.array(matrix, dtype=int), unique_labels


def matrix_to_labels(matrix, unique_labels):
    return [';'.join([unique_labels[i] for i in np.where(row)[0]]) 
            for row in matrix]




class DoseMatrix:
    """
    Construct a dosage matrix
    
    Parameters
    ----------
    labels: list
        cells' perturbation labels
        
    label_dose: list
        cells' perturbation dose values
        
    sep_pattern: str
        pattern for extracting perturbation label
        
    Examples
    --------
    >>> from DensityFlow.perturb import LabelMatrix
    >>> lb = DoseMatrix()
    >>> labels = ['p1','p2','p3','p4','c']
    >>> doses = [0.1, 0.2, 0.3, 0.4, 0]
    >>> us = lb.fit_transform(labels, doses)
    """
    def __init__(self):
        self.labels_ = None 
        
    def fit_transform(self, labels, label_dose, control_label=None, sep_pattern=r'[;_\s]'):
        mat, self.labels_ = dose_to_matrix(labels, label_dose, sep_pattern=sep_pattern)
        
        if control_label is not None:
            idx = np.where(self.labels_==control_label)[0]
            mat = np.delete(mat, idx, axis=1)
            self.labels_ = np.delete(self.labels_, idx)
            
        return mat 
    

@njit(parallel=True)
def _numba_fill_matrix(dose_matrix, cell_indices, label_indices, label_doses):
    """Numba 加速的矩阵填充函数"""
    for i in range(len(label_indices)):
        dose_matrix[cell_indices[i], label_indices[i]] = label_doses[i]

def dose_to_matrix(labels, label_dose, sep_pattern=r'[;_\-\s]', all_labels=None):
    """
    使用 Numba 的终极加速版本（需预先安装 numba）
    """
    labels_unified = [re.sub(sep_pattern, ';', label) for label in labels]
    if type(label_dose[0]) == str:
        label_dose_unified = [re.sub(sep_pattern, ';', ld) for ld in label_dose]
    else:
        label_dose_unified = label_dose
    all_unique_labels = sorted(set(chain(*[label.split(';') for label in labels_unified])))
    
    if all_labels is None:
        #all_labels = sorted(set().union(labels))
        all_labels = all_unique_labels
    
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    n_samples = len(labels)
    n_labels = len(all_labels)
    dose_matrix = np.zeros((n_samples, n_labels), dtype=np.float64)
    
    # 预处理为 Numba 兼容格式
    cell_indices = []
    label_indices = []
    label_doses = []
    for i, label_ in enumerate(labels_unified):
        if ';' in label_:
            labels_ = label_.split(';')
            if type(label_dose_unified[i])==str:
                if ';' in label_dose_unified[i]:
                    label_dose_ = label_dose_unified[i].split(';')
                else:
                    label_dose_ = [label_dose_unified[i]] * len(labels_)
            else:
                label_dose_ = [label_dose_unified[i]] * len(labels_)
                
            for j, label in enumerate(labels_):
                cell_indices.append(i)
                label_indices.append(label_to_idx[label])
                label_doses.append(float(label_dose_[j]))
        else:
            label = label_
            cell_indices.append(i)
            label_indices.append(label_to_idx[label])
            label_doses.append(float(label_dose[i]))
    
    # 调用 Numba 加速函数
    _numba_fill_matrix(dose_matrix, cell_indices, label_indices, label_doses)
    return dose_matrix,np.array(all_labels)