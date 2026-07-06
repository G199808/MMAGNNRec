# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserItemGCN(nn.Module):
    """
    轻量级解耦型 LightGCN 图卷积传播层
    """

    def __init__(self, num_layers=2):
        super(UserItemGCN, self).__init__()
        self.num_layers = num_layers

    def forward(self, norm_adj_matrix, user_emb, item_emb):
        # 垂直拼接用户与商品的状态矩阵 [N + M, d]
        h = torch.cat([user_emb, item_emb], dim=0)
        all_layers_embeddings = [h]

        # 迭代多层稀疏矩阵乘法进行邻域聚合
        for _ in range(self.num_layers):
            h = torch.sparse.mm(norm_adj_matrix, h)
            all_layers_embeddings.append(h)

        # 层聚合操作：计算所有层表征的算术平均值
        final_embeddings = torch.mean(torch.stack(all_layers_embeddings, dim=0), dim=0)
        num_users = user_emb.shape[0]

        # 重新分割回用户和商品
        return final_embeddings[:num_users], final_embeddings[num_users:]


class MultimodalAttentionFusion(nn.Module):
    """
    动态注意力晚融合模块：对图传播后的独立视角特征(ID, 视觉, 文本)进行实例级加权融合
    """

    def __init__(self, embedding_dim):
        super(MultimodalAttentionFusion, self).__init__()
        # 统一维度上的双层感知机打分网络
        self.attn_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, id_emb, v_emb, t_emb):
        # 堆叠三种特征：形状为 [3, N, embedding_dim]
        stacked_features = torch.stack([id_emb, v_emb, t_emb], dim=0)

        # 计算注意力分数与归一化权重 [3, N, 1]
        scores = self.attn_net(stacked_features)
        attn_weights = F.softmax(scores, dim=0)

        # 加权求和，得到最终的联合表征 [N, embedding_dim]
        fused_emb = torch.sum(attn_weights * stacked_features, dim=0)
        return fused_emb, attn_weights