import numpy as np


class New_Event():
    def __init__(self, event_event, event_group, new_event_event, event_cluster, noise_num):
        self.cluster_center = dict(zip(event_cluster['id'].values.tolist(),
                                       event_cluster['center'].values.tolist()))  # 聚类中心向量字典
        self.cluster_threshold = dict(zip(event_cluster['id'].values.tolist(),
                                       event_cluster['threshold'].values.tolist()))  # 聚类相似度阈值字典
        self.cluster_group = dict(zip(event_cluster['id'].values.tolist(),
                                       event_cluster['group'].values.tolist()))  # 聚类-类别对照关系字典
        self.group_vector = dict(zip(event_group['id'].values.tolist(),
                                       event_group['label_bert'].values.tolist()))  # 各事件类别所有关键词词向量列表字典
        self.train_data = event_event  # 训练数据集
        self.test_data = new_event_event  # 测试数据集
        self.keywords_list = new_event_event['keyword'].values.tolist()  #所有测试数据的关键词列表
        self.feature = new_event_event['word_embedding'].values.tolist()  # 测试数据代表向量
        self.group_result = []  # 测试数据分类结果
        self.cluster_result = []  # 测试数据聚类结果
        #self.sim_result = []  # 测试数据相似度结果
        self.noise_num = noise_num  # 现有噪点数据量

    def first_classify(self):
        df = self.train_data.drop_duplicates(subset=['keyword'], keep='first')  # 获取训练数据的关键词、聚类和事件类别结果
        train_keyword_list = df['keyword'].values.tolist()
        train_group_num = df['group'].values.tolist()
        train_cluster = df['cluster'].values.tolist()
        for event in self.keywords_list:  # 遍历新数据的关键词
            group_tmp = []
            sim_tmp = []
            cluster_tmp = []
            for i, j, z in zip(train_keyword_list, train_group_num, train_cluster):
                if str(i) == str(event):  # 若新数据与训练数据中某条数据的关键词列表相同
                    group_tmp.append(j)  # 按照训练数据事件类别分类
                    sim_tmp.append(1)
                    cluster_tmp.append(z)  # 按照训练数据聚类结果聚类
            if group_tmp and cluster_tmp:
                self.group_result.append(dict(zip(group_tmp,sim_tmp)))  # 记录新数据的所有分类结果
                self.cluster_result.append(cluster_tmp)  # 记录新数据的所有聚类结果
            else:
                self.group_result.append([])  # 若未找到相同数据，分类结果为空
                #self.sim_result.append([])  # 若未找到相同数据，相似度结果为空
                self.cluster_result.append([])  # 若未找到相同数据，聚类结果为空
        return

    #余弦相似度计算
    def cosSim(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    # Attention机制
    def attention(self, query):
        group_vec = {}
        for group,vector_list in self.group_vector.items():  # 遍历事件类别向量字典
            weight = []
            result = []
            for vector in vector_list:  # 遍历其中每个关键词的词向量
                wei = self.cosSim(np.array(query), np.array(vector))  # 以与query的相似度作为向量权重
                weight.append(wei)
            for m in range(768):
                temp = 0
                for n in range(len(vector_list)):
                    temp += vector_list[n][m] * weight[n]  # 根据每个关键词的词向量和权重，加权平均生成事件类别的代表向量
                result.append(temp)
            group_vec[group] = result  # 记录所有事件类别的代表向量
        return group_vec

    # 新数据分类
    def event_classify(self):
        # 先进行预分类
        self.first_classify()

        # 未分类成功数据分类
        for index, group in enumerate(self.group_result):
            if group:
                continue
            new = self.feature[index]  # 未分类成功数据的代表向量
            flag = False
            tmp = []
            tmp1 = []
            for cluster,center in self.cluster_center.items():  # 遍历聚类中心字典
                s = self.cosSim(np.array(new), np.array(center))  # 与聚类中心计算相似度
                if s < self.cluster_threshold[cluster]:  # 与聚类中心相似度不符合阈值
                    continue
                else:  # 与聚类中心相似度符合阈值
                    flag = True
                    tmp.append(tuple([self.cluster_group[cluster], s]))  # 记录与某一类别的相似度大小
                    tmp1.append(tuple([cluster, s]))  # 记录与某一聚类的相似度大小
            if flag == True:  # 能找到所属聚类
                tmp = tmp.sort(key=lambda x: x[1], reverse = True)  # 将分类结果按相似度大小降序排序
                tmp1 = tmp1.sort(key=lambda x: x[1], reverse = True)  # 将聚类结果按相似度大小降序排序

                temp = [x[0] for x in tmp[0:5]]
                temp1 = [x[0] for x in tmp1[0:5]]
                temp2 = [x[1] for x in tmp[0:5]]
                self.group_result[index] = dict(zip(temp,temp2))
                #self.sim_result[index] = temp2
                self.cluster_result[index]=temp1
            else:
                # 未找到所属聚类，按噪点数据分类处理
                self.noise_num += 1
                group_num, sim = self.noise_process(np.array(new))  # 对噪点数据进行分类
                self.group_result[index] = dict(zip(group_num,sim))  # 记录噪点数据的分类结果
                #self.sim_result[index] = sim
                self.cluster_result[index].append(-1)  # 记录噪点数据的聚类结果（-1）
        self.test_data['group_classify'] = self.group_result
        self.test_data['cluster'] = self.cluster_result  # 记录所有新数据的分类和聚类结果
        #self.test_data['sim'] = self.cluster_result  # 记录所有新数据的分类和聚类结果
        return self.test_data, self.noise_num

    # 计算相似度及分类
    def cul_simlarity(self, point, group_vec, group_threshold):
        groups_result = {}  # 记录分类结果及相似度
        res = []
        sim = []
        for i, j in group_vec.items():
            s = self.cosSim(np.array(point), np.array(j))  # 计算噪点数据向量与事件类别向量的相似度
            if s >= group_threshold:
                groups_result[i] = s  # 记录符合相似度阈值的分类结果
        if groups_result.keys():
            tmp = dict(sorted(groups_result.items(), key=lambda x: x[1], reverse=True))  # 将分类结果按照相似度大小降序排序
            num = 0
            for i, j in tmp.items():
                num += 1
                if num <= 5:
                    res.append(i)  # 记录相似度大小前五的分类结果
                    sim.append(j)  # 记录相似度
                else:
                    break
        return res, sim

    #噪点数据相似度衡量及分类
    def noise_process(self, noise_point):
        group_vec = self.attention(noise_point)  # 使用Attention机制，根据噪点数据获取所有事件类别的代表向量
        group_threshold = 0.8  # 设定分类相似度阈值
        groups_result, sim_result = self.cul_simlarity(noise_point, group_vec, group_threshold)  # 对噪点数据分类
        return groups_result, sim_result
