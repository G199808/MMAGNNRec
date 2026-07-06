# -*- coding: utf-8 -*-
import torch


def evaluate_user_recommendations(model, norm_adj_matrix, raw_visual, raw_textual, test_user_id=0, k=20):
    """
    离线多指标评测：计算指定用户的全库 Top-K 评分推荐列表
    """
    model.eval()
    with torch.no_grad():
        # 1. 干净的前向传播获取固定表征
        final_users, final_items, (u_attn, i_attn) = model(norm_adj_matrix, raw_visual, raw_textual)

        # 2. 目标用户的特征向量
        u_emb = final_users[test_user_id]

        # 3. 计算对系统中全量商品的内积预测得分
        all_item_scores = torch.matmul(u_emb, final_items.T)

        # 4. 排序并检索最高的前 K 个推荐商品
        topk_scores, topk_indices = torch.topk(all_item_scores, k=k)

    return topk_indices.cpu().numpy(), topk_scores.cpu().numpy()