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

class group_vector_matrix(object):
    def __init__(self, keyword_dict):
        self.group_name = keyword_dict.group_name  # 事件类别文本描述字典
        self.self_dict = keyword_dict.self_dict  # 自定义词典
        self.keyword_vector = keyword_dict.keyword_vector  # 关键词词典中关键词和词向量的字典
        self.group_keyword = {}  # 事件类别文本描述的关键词的字典
        self.group_vector = {}  # 事件类别文本描述的关键词词向量的字典

    # 事件类别文本描述进行分词
    def word_cut(self):
        jieba.load_userdict("self_dict.txt")  # 加载自定义词典
        for i, j in enumerate(self.group_name):
            self.group_keyword[i] = []
            for word in jieba.lcut(j):
                num = re.compile(r'((-?\d+)(\.\d+)?)')  # 识别数字
                character = re.compile(r'\W+')  # 识别符号
                if character.match(word) or num.match(word):  # 去除数字和符号
                    continue
                if word not in self.group_keyword[i]:
                    self.group_keyword[i].append(word.casefold())  # 记录各事件类别的分词结果
        return

    # 获取各个事件类别文本描述的所有分词词向量
    def get_vector_dict(self):
        self.word_cut()  # 事件类别文本描述进行分词
        for group, seg_list in self.group_keyword.items():
            vector = []
            for word in seg_list:
                vector.append(self.keyword_vector[word])  # 将分词转化为词向量
            self.group_vector[group]=vector  # 记录每个事件类别的所有分词词向量
        return
