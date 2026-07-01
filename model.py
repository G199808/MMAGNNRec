import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. 升级版模型架构 (融合了 Late Fusion 与 对比学习)
# =====================================================================

class MultimodalAttentionFusion(nn.Module):
    """
    升级后的注意力融合模块：对图传播后的多模态特征进行动态加权融合
    """

    def __init__(self, embedding_dim):
        super(MultimodalAttentionFusion, self).__init__()
        # 直接在统一维度上进行注意力打分
        self.attn_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, id_emb, v_emb, t_emb):
        # 将三种特征堆叠 [3, N, dim]
        stacked_features = torch.stack([id_emb, v_emb, t_emb], dim=0)
        # 计算注意力分数
        scores = self.attn_net(stacked_features)  # [3, N, 1]
        attn_weights = F.softmax(scores, dim=0)  # [3, N, 1]

        # 加权求和得到最终融合特征 [N, dim]
        fused_emb = torch.sum(attn_weights * stacked_features, dim=0)
        return fused_emb, attn_weights


class UserItemGCN(nn.Module):
    """
    标准的 LightGCN 图传播层
    """

    def __init__(self, num_layers=2):
        super(UserItemGCN, self).__init__()
        self.num_layers = num_layers

    def forward(self, norm_adj_matrix, user_emb, item_emb):
        h = torch.cat([user_emb, item_emb], dim=0)
        all_layers_embeddings = [h]

        for _ in range(self.num_layers):
            h = torch.sparse.mm(norm_adj_matrix, h)
            all_layers_embeddings.append(h)

        final_embeddings = torch.mean(torch.stack(all_layers_embeddings, dim=0), dim=0)
        num_users = user_emb.shape[0]
        return final_embeddings[:num_users], final_embeddings[num_users:]


class MMAGNNRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, visual_dim, textual_dim, num_gcn_layers=2):
        super(MMAGNNRecommender, self).__init__()
        self.num_users = num_users

        # --- 1. ID 嵌入 ---
        self.user_id_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(num_items, embedding_dim)

        # --- 2. 模态特征投影矩阵 ---
        self.v_projection = nn.Linear(visual_dim, embedding_dim)
        self.t_projection = nn.Linear(textual_dim, embedding_dim)

        # 为用户也初始化模态偏好嵌入 (用于构建模态独立的图传播)
        self.user_v_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_t_embedding = nn.Embedding(num_users, embedding_dim)

        # --- 3. 核心模块 ---
        self.gnn = UserItemGCN(num_layers=num_gcn_layers)
        self.attention_fusion = MultimodalAttentionFusion(embedding_dim)

        # 参数初始化
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.user_v_embedding.weight)
        nn.init.xavier_uniform_(self.user_t_embedding.weight)

    def forward(self, norm_adj_matrix, raw_visual, raw_textual):
        # 1. 投影多模态特征到公共空间
        item_v_emb = self.v_projection(raw_visual)
        item_t_emb = self.t_projection(raw_textual)

        user_id_emb = self.user_id_embedding.weight
        item_id_emb = self.item_id_embedding.weight
        user_v_emb = self.user_v_embedding.weight
        user_t_emb = self.user_t_embedding.weight

        # 2. 模态解耦的图传播 (Modality-Aware Propagation)
        # ID 视角传播
        final_u_id, final_i_id = self.gnn(norm_adj_matrix, user_id_emb, item_id_emb)
        # 视觉视角传播
        final_u_v, final_i_v = self.gnn(norm_adj_matrix, user_v_emb, item_v_emb)
        # 文本视角传播
        final_u_t, final_i_t = self.gnn(norm_adj_matrix, user_t_emb, item_t_emb)

        # 3. 提取用于对比学习的商品模态特征 (保存以供 Loss 计算)
        self.item_v_propagated = final_i_v
        self.item_t_propagated = final_i_t

        # 4. 级联注意力融合 (Late Fusion)
        final_user_emb, u_attn = self.attention_fusion(final_u_id, final_u_v, final_u_t)
        final_item_emb, i_attn = self.attention_fusion(final_i_id, final_i_v, final_i_t)

        return final_user_emb, final_item_emb, i_attn

    def compute_bpr_loss(self, users, pos_items, neg_items, final_user_emb, final_item_emb):
        """标准 BPR 排序损失"""
        u_e = final_user_emb[users]
        pos_i_e = final_item_emb[pos_items]
        neg_i_e = final_item_emb[neg_items]

        pos_scores = torch.sum(u_e * pos_i_e, dim=-1)
        neg_scores = torch.sum(u_e * neg_i_e, dim=-1)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = (u_e.norm(2).pow(2) + pos_i_e.norm(2).pow(2) + neg_i_e.norm(2).pow(2)) / len(users)
        return bpr_loss, reg_loss

    def compute_ssl_loss(self, items, temp=0.2):
        """
        跨模态对比学习损失 (InfoNCE)
        拉近同一个商品的图结构视觉特征与文本特征
        """
        # 仅对当前 Batch 的商品进行对比学习，减少显存消耗
        v_e = self.item_v_propagated[items]
        t_e = self.item_t_propagated[items]

        # 归一化
        v_e = F.normalize(v_e, dim=-1)
        t_e = F.normalize(t_e, dim=-1)

        # 正样本得分 (同一个商品的 V 和 T)
        pos_score = torch.sum(v_e * t_e, dim=-1)
        pos_score = torch.exp(pos_score / temp)

        # 所有样本得分 (当前 Batch 内所有 V 和 T 的组合)
        ttl_score = torch.matmul(v_e, t_e.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)

        # InfoNCE Loss
        ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
        return ssl_loss


# =====================================================================
# 2. 真实数据读取与训练流程 (已针对 A800-80GB 及特定数据集优化)
# =====================================================================
if __name__ == '__main__':
    # -------------------------------------------------------------
    # 【核心参数修改区】 根据上图切换数据集时，仅需修改这三个数值即可
    # -------------------------------------------------------------
    # 当前设定为：Baby 数据集
    num_users = 19445  # 对应表格中的 # Users
    num_items = 7050  # 对应表格中的 # Items
    num_interactions = 139110  # 对应表格中的 # Interactions

    # 模态原始特征维度
    visual_dim = 4096  # 视觉特征明确为 4096 维
    textual_dim = 1024  # 文本特征通过 sentence-transformer 提炼为 1024 维
    embedding_dim = 64  # 推荐隐层表示维度 (常设为 64 或 128)

    # 2.1 加载文本特征文件
    try:
        real_textual_feats = np.load('text_feat.npy')
        real_textual_feats = torch.tensor(real_textual_feats, dtype=torch.float32)
        print(f"成功加载真实文本特征！矩阵形状: {real_textual_feats.shape}")
        assert real_textual_feats.shape == (num_items, textual_dim), "文本特征文件的数量或维度不匹配！"
    except (FileNotFoundError, AssertionError):
        print(f"未找到或无法匹配 text_feat.npy，使用 ({num_items}, {textual_dim}) 模拟文本特征。")
        real_textual_feats = torch.randn(num_items, textual_dim)

    # 2.2 加载视觉特征文件
    try:
        real_visual_feats = np.load('visual_feat.npy')
        mock_visual_feats = torch.tensor(real_visual_feats, dtype=torch.float32)
        print(f"成功加载真实视觉特征！矩阵形状: {mock_visual_feats.shape}")
        assert mock_visual_feats.shape == (num_items, visual_dim), "视觉特征文件的数量或维度不匹配！"
    except (FileNotFoundError, AssertionError):
        print(f"未找到或无法匹配 visual_feat.npy，使用 ({num_items}, {visual_dim}) 模拟视觉特征。")
        mock_visual_feats = torch.randn(num_items, visual_dim)

    # 2.3 依据对应数据集的实际交互边构建稀疏邻接矩阵
    total_nodes = num_users + num_items
    # 真实训练时请使用真实的用户-商品对索引 (形如 [[u1, u2, ...], [i1, i2, ...]]) 代替下面的 randint
    indices = torch.randint(0, total_nodes, (2, num_interactions))
    values = torch.ones(num_interactions)
    norm_adj_matrix = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes)).coalesce()

    # 2.4 初始化模型并向 A800 GPU 搬运
    model = MMAGNNRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        visual_dim=visual_dim,
        textual_dim=textual_dim,
        num_gcn_layers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    real_textual_feats = real_textual_feats.to(device)
    mock_visual_feats = mock_visual_feats.to(device)
    norm_adj_matrix = norm_adj_matrix.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 2.5 增大 Batch Size 以发挥 A800-80GB 的吞吐性能
    # 2048可以显著缩短单轮 Epoch 的训练时间
    batch_size = 2048
    sample_users = torch.randint(0, num_users, (batch_size,)).to(device)
    sample_pos_items = torch.randint(0, num_items, (batch_size,)).to(device)
    sample_neg_items = torch.randint(0, num_items, (batch_size,)).to(device)

    # 2.6 前向传播与计算损失
    model.train()
    optimizer.zero_grad()

    # 获取融合后的最终表示
    final_users, final_items, attn_weights = model(norm_adj_matrix, mock_visual_feats, real_textual_feats)

    # 计算主任务损失 (BPR + L2正则化)
    bpr_loss, reg_loss = model.compute_bpr_loss(sample_users, sample_pos_items, sample_neg_items, final_users,
                                                final_items)

    # 计算自监督辅助任务损失 (对比学习)
    ssl_loss = model.compute_ssl_loss(sample_pos_items, temp=0.2)

    # 综合多任务损失
    ssl_weight = 0.05
    total_loss = bpr_loss + 1e-4 * reg_loss + ssl_weight * ssl_loss

    total_loss.backward()
    optimizer.step()


    print(f"当前运行数据集规模 -> 用户数: {num_users}, 商品数: {num_items}, 交互边数: {num_interactions}")
    print(f"BPR 推荐损失: {bpr_loss.item():.4f}")
    print(f"SCL 对比学习损失: {ssl_loss.item():.4f}")
    print(f"总优化 Loss: {total_loss.item():.4f}")
    print(f"当前 Batch Size: {batch_size}")