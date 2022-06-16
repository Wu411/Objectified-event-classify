from Event import Event
from new_event import New_Event
import pandas as pd
from urllib import parse
import jieba
import re
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
#获取test_event_event数据
path6 = 'D:\\毕设数据\\数据\\test_event_event.xlsx'
test_event_event = pd.read_excel(path6, sheet_name="Sheet1")

if __name__ =="__main__":

    test_events = Event(test_event_event, event_keyword, self_dict)
    test_event_event, self_dict = test_events.run()  # 获取测试数据的关键词列表和代表向量

    noise_num = len(train_event_event.loc[train_event_event['cluster'] == -1])
    new_event = New_Event(train_event_event, event_group, test_event_event, event_cluster, noise_num)
    test_event_event, noise_num = new_event.event_classify()
