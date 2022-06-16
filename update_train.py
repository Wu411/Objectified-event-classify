from cluster_text import Cluster
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


#获取event_event数据
path4 = 'D:\\毕设数据\\数据\\event_event.xlsx'
train_event_event = pd.read_excel(path4, sheet_name="Sheet1")
#获取event_cluster数据
path5 = 'D:\\毕设数据\\数据\\event_cluster.xlsx'
event_cluster = pd.read_excel(path5, sheet_name="Sheet1")


if __name__ =="__main__":

    #更新模型
    events_clusters = Cluster(train_event_event, event_cluster)
    event_cluster, train_event_event = events_clusters.run()  # 训练数据预聚类
