import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入模块
from model import MMAGNNRecommender
from data_utils import (
    load_amazon_dataset,
    build_adj_matrix,
    scipy_to_torch_sparse,
    BPRDataset,
    calculate_metrics
)

# =====================================================================
# 全局超参数配置 
# =====================================================================
DATA_NAME = 'Baby'  # 可自主切换为 'Sports' 或 'Clothing'
DATA_PATH = f'./data/{DATA_NAME}'

EMBEDDING_DIM = 64
GCN_LAYERS = 2
VISUAL_DIM = 4096
TEXTUAL_DIM = 1024
LR = 0.001
WEIGHT_DECAY = 1e-4  # 即公式中的 lambda_1
BATCH_SIZE = 2048  # 发挥 A800 算力的完美吞吐 Batch Size
SSL_TEMP = 0.2  # 对比学习温度系数 tau
SSL_WEIGHT = 0.05  # 对比学习损失权重 lambda_2
EPOCHS = 50  # 真实数据集标准轮数

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_info_nce_loss(item_v_emb, item_t_emb, batch_items, temperature=0.2):
    """
    在当前训练Mini-Batch内并行计算多模态解耦图通道间的辅助自监督 InfoNCE 损失
    """
    unique_items = torch.unique(batch_items)
    v_emb = item_v_emb[unique_items]
    t_emb = item_t_emb[unique_items]

    # L2 归一化映射到超球面上 [cite: 90]
    v_emb = F.normalize(v_emb, p=2, dim=1)
    t_emb = F.normalize(t_emb, p=2, dim=1)

    # 向量点积算其余弦相似度 [cite: 91]
    scores = torch.matmul(v_emb, t_emb.t()) / temperature  # [B_unique, B_unique]
    labels = torch.arange(v_emb.shape[0]).to(v_emb.device)  # 对角线互为正样本对 [cite: 94]

    # 交叉熵损失等价于 InfoNCE 负对数底数转化
    return F.cross_entropy(scores, labels)


def evaluate_model(model, norm_adj_matrix, raw_visual, raw_textual, train_user_set, test_user_set, num_users,
                   num_items):
    """
    全量物品排序评测协议 规避负采样偏见
    通过 GPU 矩阵高并发乘法并行计算，将 A800 显卡空闲率降到最低
    """
    model.eval()
    with torch.no_grad():
        # 前向传播：提取所有用户的最终表征与全量商品表征 [cite: 134]
        final_users, final_items, _ = model(norm_adj_matrix, raw_visual, raw_textual)

        test_users = list(test_user_set.keys())
        all_metrics = []

        eval_batch_size = 4096
        for i in range(0, len(test_users), eval_batch_size):
            batch_u = test_users[i:i + eval_batch_size]
            u_emb = final_users[batch_u]  # [B_eval, dim]

            # 点积并行爆发：计算批量用户对全城商品的初始分矩阵 [B_eval, num_items] [cite: 135]
            scores = torch.matmul(u_emb, final_items.t())

            # 严格过滤掩码：将训练集中该用户买过的商品得分写死为负无穷，使其绝不入选Top-K [cite: 126]
            for idx, u in enumerate(batch_u):
                if u in train_user_set:
                    train_items = list(train_user_set[u])
                    scores[idx, train_items] = -1e9

                    # 并行提取前 K 大的推荐项 [cite: 135]
            _, topk_indices = torch.topk(scores, k=20, dim=-1)
            topk_indices = topk_indices.cpu().numpy()

            # 载入真实测试集的 Ground Truth
            batch_ground_truth = [test_user_set[u] for u in batch_u]

            # 记录批次结果
            batch_res = calculate_metrics(topk_indices, batch_ground_truth, k_list=[10, 20])
            all_metrics.append(batch_res)

        # 汇总全局测试集均值
        global_metrics = {}
        for key in all_metrics[0].keys():
            global_metrics[key] = np.mean([bm[key] for bm in all_metrics])

        return global_metrics


def main():
    print(f"🚀 Training MMAGNNRec on REAL Dataset: {DATA_NAME} using {DEVICE}")

    # 1. 加载Amazon数据集
    train_data, train_user_set, test_user_set, num_users, num_items, raw_v, raw_t = load_amazon_dataset(DATA_PATH)

    # 2. 生成拉普拉斯稀疏图，并一步到位推入 GPU VRAM (杜绝 Epoch 内的动态传输)
    print("Building Normalized Adjacency Matrix...")
    norm_adj_coo = build_adj_matrix(num_users, num_items, train_data)
    norm_adj_matrix = scipy_to_torch_sparse(norm_adj_coo).to(DEVICE)

    raw_visual = torch.from_numpy(raw_v).to(DEVICE)
    raw_textual = torch.from_numpy(raw_t).to(DEVICE)

    # 3. 初始化优化版数据集与高并行 DataLoader
    dataset = BPRDataset(train_data, num_items, train_user_set)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # 4进程异步预加载
        pin_memory=True  # 锁页内存技术：硬件层级的 CPU-GPU 高速拷贝
    )

    # 4. 初始化模型与优化器
    model = MMAGNNRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        visual_dim=VISUAL_DIM,
        textual_dim=TEXTUAL_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_recall_20 = 0.0

    # 5. 协同多任务训练迭代圈
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_bpr, total_reg, total_ssl = 0.0, 0.0, 0.0, 0.0

        for batch_u, batch_pos, batch_neg in train_loader:
            batch_u = batch_u.to(DEVICE)
            batch_pos = batch_pos.to(DEVICE)
            batch_neg = batch_neg.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            final_users, final_items, _ = model(norm_adj_matrix, raw_visual, raw_textual)

            # 任务一：BPR 主序对损失 + L2 Regularization [cite: 108]
            bpr_loss, reg_loss = model.compute_bpr_loss(
                batch_u, batch_pos, batch_neg, final_users, final_items
            )

            # 任务二：对比学习辅助损失
            ssl_loss = compute_info_nce_loss(
                model.item_v_propagated, model.item_t_propagated, batch_pos, temperature=SSL_TEMP
            )

            # 多任务目标统一反向传播联合沉淀
            loss = bpr_loss + WEIGHT_DECAY * reg_loss + SSL_WEIGHT * ssl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bpr += bpr_loss.item()
            total_reg += reg_loss.item()
            total_ssl += ssl_loss.item()

        # 执行快速全量评测
        metrics = evaluate_model(
            model, norm_adj_matrix, raw_visual, raw_textual,
            train_user_set, test_user_set, num_users, num_items
        )

        print(f"📊 Epoch [{epoch}/{EPOCHS}] | Loss: {total_loss / len(train_loader):.4f} | "
              f"BPR: {total_bpr / len(train_loader):.4f} | SSL: {total_ssl / len(train_loader):.4f}")
        print(f"Eval: Recall@10: {metrics['recall@10']:.4f} | Recall@20: {metrics['recall@20']:.4f} | "
              f"NDCG@10: {metrics['ndcg@10']:.4f} | NDCG@20: {metrics['ndcg@20']:.4f}")

        # 记录并保存最优模型状态
        if metrics['recall@20'] > best_recall_20:
            best_recall_20 = metrics['recall@20']
            torch.save(model.state_dict(), f"best_model_{DATA_NAME}.pt")
            print("New Best Model Copied & Saved!")


if __name__ == '__main__':
    main()