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

class keyword(object):
    def __init__(self, data, keyword_weight, keyword_vector, self_dict):
        self.data = data  # 训练和测试的dataframe数据
        self.summary = data['Summary'].values.tolist()
        self.processed_summary = []  # 预处理后的summary
        self.keyword_weight = keyword_weight  # 关键词词典中关键词和权重的字典
        self.keyword_vector = keyword_vector  # 关键词词典中关键词和词向量的字典
        self.self_dict = self_dict  # 自定义词典
        self.keywords_list=[]  # 所有事件数据的关键词列表
    def process_data(self):
        for v in self.summary:
            v = parse.unquote(v)  # 解码
            v = v.casefold()
            v = v.replace("请联系业务岗处理", " ")
            v = v.replace("请联系业务岗确认", " ")
            v = v.replace("需人工介入", " ")
            v = v.replace("联系业务", " ")
            v = v.replace("请联系数据库岗处理", " ")
            v = v.replace("处理", " ")
            v = v.replace("hostname", " ")
            k = self.format_str(v, self.self_dict)  # 正则表达式处理
            k = k.replace("DATE", " ")
            k = k.replace("code", " ")
            k = k.replace("symbol", " ")
            k = k.replace("TIME", " ")
            k = k.replace("path", " ")
            k = k.replace("url", " ")
            k = k.replace("DOMAIN", " ")
            self.processed_summary.append(k)
        return
    #正则表达式处理冗余信息
    def format_str(self,tmp_txt,self_dict):
        tmp_txt = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'DATE', tmp_txt)
        tmp_txt = re.sub(r'\d{4}/\d{1,2}/\d{1,2}', 'DATE', tmp_txt)
        tmp_txt = re.sub(r'\d{1,2}/[a-zA-Z]{3}/\d{4}', 'DATE', tmp_txt)
        tmp_txt = re.sub(r'(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?', 'TIME', tmp_txt)
        tmp_txt = re.sub(r'((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}', 'ip',
                         tmp_txt)
        tmp_txt = re.sub(r'/[^\s\u4e00-\u9fa5]*\.[^\s\u4e00-\u9fa5]*', 'url', tmp_txt)
        tmp_txt = re.sub(r'\.?(/[^\s\u4e00-\u9fa5]+)+', 'path', tmp_txt)
        tmp_txt = re.sub(r'[a-zA-Z0-9_][-a-zA-Z0-9_]{0,62}(\.[a-zA-Z0-9_][-a-zA-Z0-9_]{0,62})+\.?', 'DOMAIN', tmp_txt)
        tmp_txt = re.sub(r'\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*', 'Email', tmp_txt)
        self.update_selfdict(tmp_txt, self_dict)  # 动态更新自定义词典，加入以下划线和连字符连接的词组
        tmp_txt = re.sub(r'[A-Za-z0-9]{20,}', 'code', tmp_txt)
        tmp_txt = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]{2,}', 'symbol', tmp_txt)
        return tmp_txt

    #自更新自定义词典
    def update_selfdict(self, txt, res):
        spec_words = re.findall(r'([a-zA-Z][a-zA-Z0-9]+([-_][a-zA-Z0-9]+)+)', txt)  # 识别下划线和连字符所连固定搭配动态加入自定义词典
        for i in spec_words:
            if i[0] not in res:
                res.append(i[0])
        return

    #获取某条事件关键词
    def getkeyword(self, txt):
        jieba.load_userdict("self_dict.txt")  # 加载自定义词典
        seg_list = jieba.lcut(txt)  # 对预处理后的summary分词
        words = []
        num = 0
        # 选出前十个关键词
        for k, v in self.keyword_weight.items():
            if num == 10:
                break
            if k in seg_list and k not in words:  # 找关键词词典中的词是否出现在summary中
                words.append(k)  # 记录关键词
                num += 1
        return words

    #获取所有事件数据关键词
    def run(self):
        self.process_data()
        for i in self.processed_summary:
            words = self.getkeyword(i)
            self.keywords_list.append(words)
        self.data['keywords'] = self.keywords_list  # 将关键词提取结果进行记录