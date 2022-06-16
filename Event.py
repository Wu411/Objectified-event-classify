from urllib import parse
import jieba
import re

class Event(object):
    def __init__(self, event_event, event_keyword, self_dict):
        self.event_event = event_event  # 训练和测试的dataframe数据
        self.summary = event_event['detail'].values.tolist()
        self.processed_summary = []  # 预处理后的summary
        self.keyword_weight = dict(zip(event_keyword['keyword'].values.tolist(),
                                       event_keyword['weight'].values.tolist()))  # 关键词词典中关键词权重字典
        self.keyword_vector = dict(zip(event_keyword['keyword'].values.tolist(),
                                       event_keyword['word_embedding'].values.tolist()))  # 关键词词典中关键词词向量字典
        self.self_dict = self_dict['selfdict'].values.tolist()  # 自定义词典

    # 预处理事件数据
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
        jieba.load_userdict(self.self_dict)  # 加载自定义词典
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

    # 生成事件数据的代表向量
    def to_vector(self, data, weight, vector):
        output = []
        for seg_list, wei_list, vec_list in zip(data, weight, vector):
            if seg_list:
                new = []
                for i in range(768):
                    temp = 0
                    for j in range(len(vec_list)):
                        temp += vec_list[j][i] * wei_list[j]  # 利用词向量及词权重，加权平均生成代表向量
                    new.append(temp / (len(vec_list)))
            else:
                new = [0 for i in range(768)]
            output.append(new)
        return output

    # 获取所有事件数据关键词和代表向量
    def run(self):
        # 事件数据预处理
        self.process_data()

        # 获取所有事件数据关键词
        event_keywords_list = []
        for i in self.processed_summary:
            words = self.getkeyword(i)
            event_keywords_list.append(words)
        self.event_event['keyword'] = event_keywords_list  # 将关键词提取结果进行记录

        # 对数据集中所有事件数据获取关键词词权重、词向量，生成代表向量
        res_weights = []
        res_vector = []
        for event in event_keywords_list:  # 遍历所有事件数据的关键词列表
            weights = []
            vectors = []
            if event:
                for word in event:  # 遍历关键词列表
                    weights.append(self.keyword_weight[word])  # 获取关键词的词权重
                    vectors.append(self.keyword_vector[word])  # 获取关键词的词向量
            res_weights.append(weights)
            res_vector.append(vectors)
        self.feature = self.to_vector(event_keywords_list, res_weights, res_vector)
        self.event_event['word_embedding'] = self.feature  # 记录所有事件数据生成的代表向量

        return self.event_event, self.self_dict

