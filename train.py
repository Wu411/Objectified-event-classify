from tfidf_dict import keyword_dict
from get_keyword import keyword
from get_bert import word_vector
from pre_cluster_text import cluster
from group_name_bert import group_vector_matrix
from new_event_process import new_event
import pandas as pd
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

#打开自定义词典
with open("self_dict.txt", "r", encoding='utf-8') as f1:
    self_dict = []
    for line in f1.readlines():
        word = line.strip('\n')
        self_dict.append(word)

#获取建立关键词词典数据
path1 = 'D:\\毕设数据\\数据\\副本train3_增加groupname.xlsx'
all_data = pd.read_excel(path1, sheet_name="工作表 1 - train")
#获取训练数据
path2 = 'D:\\毕设数据\\数据\\train_data.xlsx'
train_data = pd.read_excel(path2, sheet_name="Sheet1")
#获取测试数据
path3 = 'D:\\毕设数据\\数据\\test_data.xlsx'
test_data = pd.read_excel(path3, sheet_name="Sheet1")

if __name__ =="__main__":
    # 训练模型
    dictionary = keyword_dict(all_data, self_dict)
    dictionary.set_up()  # 建立关键词词典

    group_vec_dict = group_vector_matrix(dictionary)
    group_vec_dict.get_vector_dict()  #  获取事件类别文本描述的分词词向量结果

    events_keywords = keyword(train_data, dictionary.keyword_weight, dictionary.keyword_vector, dictionary.self_dict)
    events_keywords.run()  # 获取训练数据的关键词

    events_vectors = word_vector(dictionary, events_keywords)
    events_vectors.run()  # 获取训练数据代表向量

    events_clusters = cluster(events_vectors)
    events_clusters.run()  # 训练数据预聚类

    events_clusters.data.to_excel(path2, sheet_name="Sheet1")  # 记录训练数据预聚类结果

    # 测试模型
    train_data = pd.read_excel(path2, sheet_name="Sheet1")
    #dictionary = keyword_dict(all_data, self_dict)
    #dictionary.set_up()
    events_keywords = keyword(test_data, dictionary.keyword_weight, dictionary.keyword_vector, dictionary.self_dict)
    events_keywords.run()  # 获取测试数据的关键词

    events_vectors = word_vector(dictionary, events_keywords)
    events_vectors.run()  # 获取测试数据代表向量

    new_event = new_event(train_data, events_keywords, group_vec_dict.group_vector, events_vectors,
                          events_clusters.clusters_center, events_clusters.clusters_threshold,
                          events_clusters.cluster_group, events_clusters.noise_num)
    new_event.event_classify()
    new_event.test_data.to_excel(path3, sheet_name="Sheet1")