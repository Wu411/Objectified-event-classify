from group_label_process import Group
from Event import Event
from cluster_text import Cluster
import pandas as pd
import jieba
import re
from collections import defaultdict
import math
import operator
from bert_serving.client import BertClient
from urllib import parse
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


#获取event_group数据
path1 = 'D:\\毕设数据\\数据\\event_group.xlsx'
event_group = pd.read_excel(path1, sheet_name="Sheet1")
#获取event_keyword数据
path2 = 'D:\\毕设数据\\数据\\event_keyword.xlsx'
event_keyword = pd.read_excel(path2, sheet_name="Sheet1")
#获取self_dict数据
path3 = 'D:\\毕设数据\\数据\\self_dict.xlsx'
self_dict = pd.read_excel(path3, sheet_name="Sheet1")
#获取event_event数据
path4 = 'D:\\毕设数据\\数据\\event_event.xlsx'
train_event_event = pd.read_excel(path4, sheet_name="Sheet1")
#获取event_cluster数据
path5 = 'D:\\毕设数据\\数据\\event_cluster.xlsx'
event_cluster = pd.read_excel(path5, sheet_name="Sheet1")


if __name__ =="__main__":
    # 训练模型
    Group=Group(event_group, event_keyword, self_dict)
    event_group, event_keyword, self_dict = Group.run()  # 建立关键词词典,获取事件类别关键词向量

    train_events = Event(train_event_event, event_keyword, self_dict)
    train_event_event, self_dict = train_events.run()  # 获取训练数据的关键词列表和代表向量

    events_clusters = Cluster(train_event_event, event_cluster)
    event_cluster, train_event_event = events_clusters.run()  # 训练数据预聚类
