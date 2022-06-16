from cluster_text import Cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
import pandas as pd


#获取event_event数据
path1 = 'D:\\毕设数据\\数据\\event_event.xlsx'
train_event_event = pd.read_excel(path1, sheet_name="Sheet1")
#获取event_cluster数据
path2 = 'D:\\毕设数据\\数据\\event_cluster.xlsx'
event_cluster = pd.read_excel(path2, sheet_name="Sheet1")


if __name__ =="__main__":

    #噪点数据处理
    noise_event_event = train_event_event.loc[train_event_event['cluster']==-1]
    noise_process = Cluster(noise_event_event, event_cluster)
    event_cluster, noise_event_event = noise_process.run()
    train_event_event.loc[train_event_event['cluster'] == -1]['cluster'] = noise_event_event['cluster'].values.tolist()