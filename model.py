# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import UserItemGCN, MultimodalAttentionFusion

class MMAGNNRecommender(nn.Module):
    """
    MMAGNNRec 多模态注意力图神经网络推荐模型主类
    """
    def __init__(self, num_users, num_items, embedding_dim, visual_dim, textual_dim, num_gcn_layers=2):
        super(MMAGNNRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items

        # --- 1. 基础协作特征的 Lookup Table ---
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)

        # --- 2. 跨模态特征线性投影矩阵 ---
        self.v_projection = nn.Linear(visual_dim, embedding_dim)
        self.t_projection = nn.Linear(textual_dim, embedding_dim)

        # --- 3. 用户特有的多视角偏好嵌入 (保持多路结构对称) ---
        self.user_v_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_t_embedding = nn.Embedding(num_users, embedding_dim)

        # --- 4. 核心网络组件 (从 modules 导入) ---
        self.gnn = UserItemGCN(num_layers=num_gcn_layers)
        self.attention_fusion = MultimodalAttentionFusion(embedding_dim)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.user_v_embedding.weight)
        nn.init.xavier_uniform_(self.user_t_embedding.weight)

    def forward(self, norm_adj_matrix, raw_visual, raw_textual):
        # 1. 投影原始多模态特征到公共维度空间
        item_v_emb = self.v_projection(raw_visual)
        item_t_emb = self.t_projection(raw_textual)

        user_id_emb = self.user_id_embedding.weight
        item_id_emb = self.item_id_embedding.weight
        user_v_emb = self.user_v_embedding.weight
        user_t_emb = self.user_t_embedding.weight

        # 2. 模态解耦的并行图传播
        final_u_id, final_i_id = self.gnn(norm_adj_matrix, user_id_emb, item_id_emb)
        final_u_v, final_i_v = self.gnn(norm_adj_matrix, user_v_emb, item_v_emb)
        final_u_t, final_i_t = self.gnn(norm_adj_matrix, user_t_emb, item_t_emb)

        # 3. 暂存图传播后的独立模态商品表征 (供辅助任务 InfoNCE 损失计算)
        self.item_v_propagated = final_i_v
        self.item_t_propagated = final_i_t

        # 4. 实例级动态晚融合
        final_user_emb, u_attn = self.attention_fusion(final_u_id, final_u_v, final_u_t)
        final_item_emb, i_attn = self.attention_fusion(final_i_id, final_i_v, final_i_t)

        return final_user_emb, final_item_emb, (u_attn, i_attn)

    def compute_bpr_loss(self, users, pos_items, neg_items, final_users, final_items):
        """
        核心推荐任务：贝叶斯个性化排序损失 (BPR Loss) + L2正则化
        """
        u_emb = final_users[users]
        pos_i_emb = final_items[pos_items]
        neg_i_emb = final_items[neg_items]

        # 计算内积偏好预测分
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)

        # 核心排序损失
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # 精确对当前 Batch 激活的参数空间进行 L2 正则化约束
        reg_loss = (u_emb.norm(2).pow(2) + pos_i_emb.norm(2).pow(2) + neg_i_emb.norm(2).pow(2)) / len(users)

        return bpr_loss, reg_loss

    def compute_ssl_loss(self, active_items, ssl_temp=0.2):
        """
        辅助任务：跨模态自监督对比学习损失 (InfoNCE)
        """
        if len(active_items) == 0:
            return torch.tensor(0.0).to(self.item_v_propagated.device)

        # 提取当前 Batch 活跃商品的视觉与文本传播特征
        v_features = self.item_v_propagated[active_items]
        t_features = self.item_t_propagated[active_items]

        # L2 归一化投影到单位超球面上
        v_norm = F.normalize(v_features, p=2, dim=1)
        t_norm = F.normalize(t_features, p=2, dim=1)

        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(v_norm, t_norm.T) / ssl_temp

        # 正样本为对角线元素
        labels = torch.arange(len(active_items)).long().to(similarity_matrix.device)

        # 双向对比损失求均值
        loss_v_to_t = F.cross_entropy(similarity_matrix, labels)
        loss_t_to_v = F.cross_entropy(similarity_matrix.T, labels)

        return (loss_v_to_t + loss_t_to_v) / 2.0