import jieba
import re
from collections import defaultdict
import math
import operator
from bert_serving.client import BertClient

class Group(object):
    def __init__(self, event_group, event_keyword, self_dict):
        self.event_group = event_group
        self.event_keyword = event_keyword
        self.self_dict = self_dict['selfdict'].values.tolist()  # 自定义词典
        self.keyword_weight = {}  # 关键词词典中关键词权重字典
        self.keyword_vector = {}  # 关键词词典中关键词词向量字典
        self.group_name = dict(zip(event_group['id'].values.tolist(),
                                   event_group['label'].values.tolist()))  # 所有事件类别的文本描述字典
        self.group_keyword = {}  # 事件类别文本描述的关键词的字典
        self.group_vector = {}  # 事件类别文本描述的关键词词向量的字典

    # 预处理事件类别的文本描述
    def process_data(self):
        self.update_selfdict(self.group_name.values(), self.self_dict)
        jieba.load_userdict(self.self_dict)  # 加载自定义词典
        num = re.compile(r'((-?\d+)(\.\d+)?)')
        character = re.compile(r'\W+')
        for g, w in self.group_name.items():  # 遍历事件类别的文本描述数据
            l = []
            for j in jieba.lcut(w):  # 事件类别的文本描述分词
                if not character.match(j) and not num.match(j):  # 去除数字、符号
                    l.append(j.casefold())
            self.group_keyword[g] = l
        return

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
        for file in data.keys():
            for word in data[file]:
                doc_frequency[word] += 1
        '''# 计算每个词的TF值
        word_tf = {}  # 存储没个词的tf值
        for i in doc_frequency:
            word_tf[i] = doc_frequency[i] / sum(doc_frequency.values()) #sum(doc.frequency.values)'''

        # 计算每个词的IDF值
        doc_num = len(data)
        word_idf = {}  # 存储每个词的idf值
        word_doc = defaultdict(int)  # 存储包含该词的文档数
        for word in doc_frequency:
            for file in data.keys():
                if word in data[file]:
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
    def set_up_dict(self):
        feature = self.feature_select(self.group_keyword)  # 所有词的TF-IDF值
        words = []
        weights=[]
        for i in feature:
            words.append(i[0])
            weights.append(i[1])

        bc = BertClient()  # 加载Bert预训练模型
        vectors = bc.encode(words)  # 获取所有词的词向量
        for i,j,z in zip(weights, vectors, words):
            self.keyword_weight[z] = i  # 记录所有词的词权重
            self.keyword_vector[z] = j  # 记录所有词的词向量

        # 关键词词典记录在EventKeyword表中
        self.event_keyword['id'] = [i for i in range(len(words))]
        self.event_keyword['keyword'] = words
        self.event_keyword['weight'] = weights
        self.event_keyword['word_embedding'] = vectors

    def get_group_vectors(self):
        for group, seg_list in self.group_keyword.items():
            vector = []
            for word in seg_list:
                vector.append(self.keyword_vector[word])  # 将分词转化为词向量
            self.group_vector[group]=vector  # 记录每个事件类别的所有分词词向量

        # 事件类别的分词和词向量转化结果，记录在EventKeyword表中
        label_cut = []
        label_bert = []
        for group in self.event_group['id'].values.tolist():
            label_cut.append(self.group_keyword[group])
            label_bert.append(self.group_vector[group])
        self.event_group['label_cut'] = label_cut
        self.event_group['label_bert'] = label_bert

        return

    def run(self):
        self.process_data()  # 加载数据
        self.set_up_dict()  # 建立关键词词典
        self.get_group_vectors()  # 获取事件类别关键词向量
        return self.event_group, self.event_keyword, self.self_dict