from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np


class Cluster(object):
    def __init__(self, event_event, event_cluster):
        self.event_event = event_event
        self.event_cluster = event_cluster
        self.feature = event_event['word_embedding'].values.tolist()
        self.start = 0
        self.clusters_center = {}  # 聚类中心向量字典
        self.clusters_threshold = {}  # 聚类相似度阈值字典
        self.cluster_group = {}  #聚类-类别对照表
        self.noise_num = 0  # 噪点数
        self.raito = 0  # 噪点率
        self.labels = []  # 聚类结果标签
        self.silhouette_score = 0  #轮廓系数

    # 余弦相似度计算
    def cosSim(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    # 计算聚类相似度阈值
    def cul_clusters_threshold(self, center_pos, points):
        min = 1
        for i in points:  #遍历聚类中每个点
            s = self.cosSim(np.array(center_pos), np.array(i))  # 计算其与中心向量的相似度
            if s < min:
                min = s  #取最小值作为聚类相似度阈值
        return min

    # 计算聚类中心向量
    def get_center(self,X):
        kmeans = KMeans(n_clusters=1)  # 利用K-Means计算每个聚类的质心
        kmeans.fit(np.array(X))
        # 获取质心
        return (kmeans.cluster_centers_)

    def update_dbscan(self, min_eps, max_eps, eps_step, min_min_samples, max_min_samples, min_samples_step):
        eps = np.arange(min_eps, max_eps, eps_step)  # eps参数从min_eps开始到max_eps，每隔eps_step进行一次
        min_samples = np.arange(min_min_samples, max_min_samples,
                                min_samples_step)  # min_samples参数从min_min_samples开始到max_min_samples,每隔min_samples_step进行一次
        best_score = -2
        best_score_eps = 0
        best_score_min_samples = 0
        for i in eps:
            for j in min_samples:
                try:
                    DBS_clf = DBSCAN(eps=i, min_samples=j).fit(self.feature)
                    labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
                    raito_num = 0
                    for v in labels:
                        if v == -1:
                            raito_num += 1
                    raito = raito_num / len(labels)
                    # labels=-1的个数除以总数，计算噪声点个数占总数的比例
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
                    k = metrics.silhouette_score(self.feature, labels)
                    score = k - raito
                    if score > best_score:
                        best_score = score
                        best_score_eps = i
                        best_score_min_samples = j
                except:
                    DBS_clf = ''

        DBS_clf = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples).fit(self.feature)
        labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
        raito_num = 0
        for v in labels:
            if v == -1:
                raito_num += 1
        raito = raito_num / len(labels)
        # labels=-1的个数除以总数，计算噪声点个数占总数的比例
        print(best_score_eps, best_score_min_samples)
        print('噪声比:', format(raito, '.2%'))
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
        print('分簇的数目: %d' % n_clusters_)
        print("轮廓系数: %0.3f" % metrics.silhouette_score(self.feature, labels))
        return best_score_eps, best_score_min_samples

    # 聚类结果二次划分
    def Repartition(self,data):
        inx = self.start
        data['new_cluster_id'] = data['cluster']  # 设置中间变量new_cluster_id
        for cluster in set(self.labels):  # 遍历旧聚类结果
            clutser_groups_num = data.loc[data['cluster'] == cluster][
                'group_num'].drop_duplicates().values.tolist()
            for group_num in clutser_groups_num:  # 遍历一个聚类中包含的所有事件类别
                self.cluster_group[cluster]=int(group_num)  # 记录新聚类的事件类别对照关系
                res = data.loc[(data['group_num'] == group_num) & (data['cluster'] == cluster)][
                    'vector'].values.tolist()  # 对各个新聚类按照的group重新划分，一个新聚类可能形成多个新聚类
                data['new_cluster_id'].loc[(data['group_num'] == group_num) & (data['cluster'] == cluster)] \
                    = [inx for i in range(len(res))]  # 给新聚类标号
                inx += 1
                center_pos = self.get_center(res)  # 获取重新划分后的各个聚类中心
                self.clusters_center[inx] = center_pos[0]  # 记录新聚类中心向量
                threshold = self.cul_clusters_threshold(center_pos[0], res)  # 计算重新划分后的各个聚类相似度阈值
                self.clusters_threshold[inx] = threshold  # 记录新聚类相似度阈值
        data['cluster']=data['new_cluster_id'].values.tolist()  # 记录新划分的聚类结果
        data = data.drop(labels='new_cluster_id',axis=1)

        return data

    # 对训练数据集进行预聚类
    def run(self):
        best_score_eps, best_score_min_samples = self.update_dbscan(0.01, 0.2, 0.01, 2, 4, 1)  # 获取最佳参数
        DBS_clf = DBSCAN(eps=best_score_eps, min_samples=best_score_min_samples).fit(self.feature)  # 用DBSCAN聚类
        self.labels = DBS_clf.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声;
        for v in self.labels:
            if v == -1:
                self.noise_num += 1  # 记录噪点数据量
        self.raito = self.noise_num / len(self.labels)  # labels=-1的个数除以总数，计算噪声点个数占总数的比例
        self.silhouette_score=metrics.silhouette_score(self.feature, self.labels)  # 记录轮廓系数
        self.event_event['cluster']=self.labels  # 记录聚类结果
        self.event_event = self.Repartition(self.event_event)  # 记录二次划分结果

        # 记录聚类结果信息到EventCluster中
        centers = []
        groups = []
        thresholds = []
        id = self.clusters_center.keys()
        for cluster in id:
            centers.append(self.clusters_center[cluster])
            groups.append(self.cluster_group[cluster])
            thresholds.append(self.clusters_threshold[cluster])
        for i in zip(id, centers, groups, thresholds):
            inx = i[0]
            row = list(i)
            self.event_cluster.iloc[inx] = row
        self.event_cluster = self.event_cluster.drop(self.event_cluster[self.event_cluster.id > max(id)].index)

        return self.event_cluster, self.event_event
