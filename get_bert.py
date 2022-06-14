from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from urllib import parse
import jieba
import numpy as np
import re
from collections import defaultdict
import math
import operator
from bert_serving.client import BertClient

class word_vector(object):
    def __init__(self, keyword_dict, keyword):
        self.data = keyword.data  # dataframe事件数据集
        self.keyword_weight = keyword_dict.keyword_weight  # 关键词词典中关键词和权重的字典
        self.keyword_vector = keyword_dict.keyword_vector  # 关键词词典中关键词和词向量的字典
        self.event_keywords_list = keyword.keywords_list  # 数据集中每条事件数据的关键词列表
        self.feature=[]  # 数据集中每条事件数据的代表向量

    # 生成事件数据的代表向量
    def to_vector(self, data, weight, vector):
        num = 0
        output = []
        for seg_list, wei_list, vec_list in zip(data, weight, vector):
            num += 1
            if seg_list:
                new = []
                for i in range(768):
                    temp = 0
                    for j in range(len(vec_list)):
                        temp += vec_list[j][i] * wei_list[j]  #利用词向量及词权重，加权平均生成代表向量
                    new.append(temp / (len(vec_list)))
            else:
                new = [0 for i in range(768)]
            output.append(new)
        return output

    # 对数据集中所有事件数据获取关键词词权重、词向量，生成代表向量
    def run(self):
        res_weights = []
        res_vector = []
        for words in self.event_keywords_list:  # 遍历所有事件数据的关键词列表
            weights = []
            vectors = []
            for j in words:  # 遍历关键词列表
                weights.append(self.keyword_weight[j])  # 获取关键词的词权重
                vectors.append(self.keyword_vector[j])  # 获取关键词的词向量
            res_weights.append(weights)
            res_vector.append(vectors)
        self.feature = self.to_vector(self.event_keywords_list, res_weights, res_vector)
        self.data['vector'] = self.feature  # 记录所有事件数据生成的代表向量