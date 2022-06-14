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

class keyword_dict(object):
    def __init__(self, data, self_dict):
        self.group_name = data['group_name'].dropna().drop_duplicates().values.tolist()  # 所有事件类别的文本描述
        self.self_dict = self_dict  # 自定义词典
        self.keyword_weight = {}  # 关键词词典中关键词和权重的字典
        self.keyword_vector = {}  # 关键词词典中关键词和词向量的字典

    # 预处理事件类别的文本描述
    def process_data(self):
        self.update_selfdict(self.group_name, self.self_dict)
        jieba.load_userdict("self_dict.txt")  # 加载自定义词典
        dataset = []
        num = re.compile(r'((-?\d+)(\.\d+)?)')
        character = re.compile(r'\W+')
        for w in self.group_name:  # 遍历事件类别的文本描述数据
            l = []
            for j in jieba.lcut(w):  # 事件类别的文本描述分词
                if not character.match(j) and not num.match(j):  # 去除数字、符号
                    l.append(j.casefold())
            dataset.append(l)
        return dataset

    # 自更新自定义词典
    def update_selfdict(self, txt_list, res):  # 将下划线和连字符所连固定搭配动态加入自定义词典
        for txt in txt_list:
            spec_words = re.findall(r'([a-zA-Z][a-zA-Z0-9]+([-_][a-zA-Z0-9]+)+)', txt)
            for i in spec_words:
                if i[0] not in res:
                    res.append(i[0])
        return

    # 计算词权重IDF
    def feature_select(self,data):
        # 总词频统计
        doc_frequency = defaultdict(int)  # 记录每个词出现的次数，可以把它理解成一个可变长度的list，只要你索引它，它就自动扩列
        for file in data:
            for word in file:
                doc_frequency[word] += 1
        ''''# 计算每个词的TF值
        word_tf = {}  # 存储没个词的tf值
        for i in doc_frequency:
            word_tf[i] = doc_frequency[i] / sum(doc_frequency.values()) #sum(doc.frequency.values)'''

        # 计算每个词的IDF值
        doc_num = len(data)
        word_idf = {}  # 存储每个词的idf值
        word_doc = defaultdict(int)  # 存储包含该词的文档数
        for word in doc_frequency:
            for file in data:
                if word in file:
                    word_doc[word] += 1
        # word_doc和doc_frequency的区别是word_doc存储的是包含这个词的文档数，即如果一个文档里有重复出现一个词则word_doc < doc_frequency
        for word in doc_frequency:
            word_idf[word] = math.log(doc_num / (word_doc[word] + 1))

        # 计算每个词的TF*IDF的值
        word_tf_idf = {}
        for word in doc_frequency:
            word_tf_idf[word] = word_idf[word]  # * word_tf[word]

        # 对字典按值由大到小排序
        dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
        return dict_feature_select

    # 建立关键词词典
    def set_up(self):
        txt_list = self.process_data()  # 加载数据
        feature = self.feature_select(txt_list)  # 所有词的TF-IDF值
        words = []
        weights=[]
        for i in feature:
            words.append(i[0])
            weights.append(i[1])

        bc = BertClient()  # 加载Bert预训练模型
        vector = bc.encode(words)  # 获取所有词的词向量
        for i,j,z in zip(weights, vector, words):
            self.keyword_weight[z] = i  # 记录所有词的词权重
            self.keyword_vector[z] = j  # 记录所有词的词向量