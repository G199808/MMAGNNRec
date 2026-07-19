import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


def load_amazon_dataset(data_path):
    """
    Amazon多模态推荐数据集 (支持 .inter 交互文件格式)
    """
    train_data = []
    train_user_set = {}
    test_user_set = {}

    num_users = 0
    num_items = 0

    # 1. 解析训练集交互记录
    train_path = os.path.join(data_path, 'train.inter')
    with open(train_path, 'r', encoding='utf-8') as f:
        # 跳过表头 (例如 user_id:token	item_id:token)
        next(f)
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            # 支持 tab键 或 空格 分隔
            parts = list(map(int, line_str.replace('\t', ' ').split()))
            if len(parts) < 2:
                continue
            u = parts[0]
            items = parts[1:]

            if u not in train_user_set:
                train_user_set[u] = set()
            train_user_set[u].update(items)

            for i in items:
                train_data.append((u, i))
                num_users = max(num_users, u + 1)
                num_items = max(num_items, i + 1)

    # 2. 解析测试集交互记录
    test_path = os.path.join(data_path, 'test.inter')
    with open(test_path, 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = list(map(int, line_str.replace('\t', ' ').split()))
            if len(parts) < 2:
                continue
            u = parts[0]
            items = parts[1:]

            if u not in test_user_set:
                test_user_set[u] = set()
            test_user_set[u].update(items)

            num_users = max(num_users, u + 1)
            for i in items:
                num_items = max(num_items, i + 1)

    # 3. 读取预提取的高维模态特征
    visual_features = np.load(os.path.join(data_path, 'visual_emb.npy')).astype(np.float32)  # [N_i, 4096]
    textual_features = np.load(os.path.join(data_path, 'text_emb.npy')).astype(np.float32)  # [N_i, 1024]

    num_items = max(num_items, visual_features.shape[0], textual_features.shape[0])

    print(f"Dataset Loaded Successfully from {data_path}")
    print(f"Users: {num_users}, Items: {num_items}, Train Interactions: {len(train_data)}")

    return train_data, train_user_set, test_user_set, num_users, num_items, visual_features, textual_features


def build_adj_matrix(num_users, num_items, train_data):
    """
    构建拉普拉斯归一化双向图邻接矩阵
    """
    R = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    for u, i in train_data:
        R[u, i] = 1.0
    R = R.tocsr()

    # 拼接全图双向 bipartite 邻接矩阵
    adj_mat = sp.bmat([[None, R], [R.T, None]], format='csr')

    # 执行拉普拉斯对称归一化
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
    return norm_adj.tocoo()


def scipy_to_torch_sparse(sp_mat):
    """将 Scipy 稀疏矩阵高速转换为 PyTorch 密集/稀疏张量接口"""
    samples = sp_mat.tocoo()
    indices = torch.from_numpy(np.vstack((samples.row, samples.col)).astype(np.int64))
    values = torch.from_numpy(samples.data)
    shape = torch.Size(samples.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class BPRDataset(Dataset):
    """
    针对真实数据集优化的BPR负采样Dataset
    严格确保采样的负样本 item 绝不在该用户的历史训练交互集合中
    """

    def __init__(self, train_data, num_items, train_user_set):
        self.train_data = train_data
        self.num_items = num_items
        self.train_user_set = train_user_set

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        u, i = self.train_data[idx]
        while True:
            j = np.random.randint(0, self.num_items)  # 随机采样
            if j not in self.train_user_set[u]:  # 严格校验：确保用户没有买过j
                break
        return u, i, j


def calculate_metrics(topk_items, test_ground_truth, k_list=[10, 20]):
    """
    精确计算推荐系统的核心全量评测指标 Recall@K 和 NDCG@K
    """
    results = {f'recall@{k}': [] for k in k_list}
    results.update({f'ndcg@{k}': [] for k in k_list})

    for idx, truth in enumerate(test_ground_truth):
        if len(truth) == 0:
            continue
        pred = topk_items[idx]
        for k in k_list:
            hit_items = [1 if item in truth else 0 for item in pred[:k]]

            # 计算 Recall
            hits = sum(hit_items)
            recall = hits / len(truth)
            results[f'recall@{k}'].append(recall)

            # 计算 NDCG
            dcg = 0.0
            for i, hit in enumerate(hit_items):
                if hit:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(truth)))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            results[f'ndcg@{k}'].append(ndcg)

    return {metric: np.mean(values) for metric, values in results.items()}