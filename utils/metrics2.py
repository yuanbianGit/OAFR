import torch
import numpy as np
import os
from utils.reranking import re_ranking
import os.path as osp
from utils.rankResults import visualize_ranked_results

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def euclidean_distance_weight(x, y, w_x, w_y):
    """
        x (part_num, m, d) [tensor(m,d)]
        y (part_num, n, d)
        w_x (part_num, m) [tensor(m)]
        w_y (part_num, n)
    """
    dis = torch.ones((len(x), len(x[0]), len(y[0])))
    w = torch.ones((len(x), len(x[0]), len(y[0])))
    for i in range(int(len(x))):
        eu_dist = euclidean_distance(x[i], y[i])
        wx = w_x[i].reshape((len(w_x[i]), 1)).repeat((1, len(w_y[i])))
        wy = w_y[i].reshape((1, len(w_y[i]))).repeat((len(w_x[i]), 1))
        w[i] = torch.mul(wx, wy)
        dis[i] = torch.mul(torch.tensor(eu_dist), w[i])
    dis_sum = torch.sum(dis, dim=0, keepdim=False)
    w_sum = torch.sum(w, dim=0, keepdim=False)
    distance = dis_sum/w_sum  #除以这个加权是要对距离进行归一化，因为可能有些计算了两块的距离，有些计算了三块的距离！

    return distance.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """

    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    # g_pids[indices] 相当于gallary中的pid，按相似度最前排序，行id为行人的id，(q_num,g_num)
    # q_pids[:, np.newaxis] (q_num,1)
    # matches(q_num,g_num)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP



class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []


        self.pids_local = []
        self.camids_local = []


        self.g_feats = []
        self.l_feats = []
        self.weight = []

        self.l_feats_w  = []
        self.weight_w  = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def update_w(self, output):  # called once for each batch
        g_w, l_h, wei_h, l_w, wei_w,pid, camid = output

        self.g_feats.append(g_w.cpu())  #list[(B,L),(B,L),(B,L)...]
        self.l_feats.append(l_h.cpu())  #list[(4,B,L),(4,B,L)...]
        self.weight.append(wei_h.cpu())   #lsit[(4,B),(4,B)]  #是weight的hard label

        self.l_feats_w.append(l_w.cpu())
        self.weight_w.append(wei_w.cpu())   #是weight的softlabel

        self.pids_local.extend(np.asarray(pid))
        self.camids_local.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    def compute_w(self):  # called after each epoch
        g_feats = torch.cat(self.g_feats, dim=0)  #[N,L]
        l_feats = torch.cat(self.l_feats, dim=1)  #[7, N, L]
        feats_w = torch.cat(self.weight, dim=1)   #[7, N]
        feats_visibility = torch.cat(self.weight_w, dim=1)

        if self.feat_norm:
            print("The test feature is normalized")
            g_feats = torch.nn.functional.normalize(g_feats, dim=1, p=2)  # along channel

            # local_feat = [torch.nn.functional.normalize(self.l_w[i], axis=-1) for i in range(len(self.l_w))]
        # query
        # 这里可以设置只算global或者局部的特征！
        q_gf = g_feats[:self.num_query]
        q_lf = l_feats[:, :self.num_query, :]
        q_w = feats_w[:, :self.num_query]
        q_vis = feats_visibility[:, :self.num_query]
        q_pids = np.asarray(self.pids_local[:self.num_query])
        q_camids = np.asarray(self.camids_local[:self.num_query])
        # gallery
        g_qf = g_feats[self.num_query:]
        g_lf = l_feats[:, self.num_query:, :]
        g_w = feats_w[:, self.num_query:]
        g_vis = feats_w[:, self.num_query:]
        g_pids = np.asarray(self.pids_local[self.num_query:])
        g_camids = np.asarray(self.camids_local[self.num_query:])
        if self.reranking:
            # print('=> Enter reranking')
            # # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            # distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            pass
        else:

            # distmat = euclidean_distance_weight(q_lf, g_lf, q_w, g_w)
            distmat = euclidean_distance_weight(q_lf, g_lf, q_vis, g_vis)

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids

    def compute_featRecovery(self,k_num=8,first_dist = 'global', recover_method='hard_label'):  # called after each epoch
        g_feats = torch.cat(self.g_feats, dim=0)  #[N,L]
        l_feats = torch.cat(self.l_feats, dim=1)  #[7, N, L]
        feats_w = torch.cat(self.weight, dim=1)   #[7, N]
        feats_visibility = torch.cat(self.weight_w, dim=1)

        if self.feat_norm:
            print("The test feature is normalized")
            g_feats = torch.nn.functional.normalize(g_feats, dim=1, p=2)  # along channel

            # local_feat = [torch.nn.functional.normalize(self.l_w[i], axis=-1) for i in range(len(self.l_w))]
        # query
        # 这里可以设置只算global或者局部的特征！
        q_gf = g_feats[:self.num_query]
        q_lf = l_feats[:, :self.num_query, :]
        q_w = feats_w[:, :self.num_query]
        q_vis = feats_visibility[:, :self.num_query]
        q_pids = np.asarray(self.pids_local[:self.num_query])
        q_camids = np.asarray(self.camids_local[:self.num_query])
        # gallery
        g_qf = g_feats[self.num_query:]
        g_lf = l_feats[:, self.num_query:, :]
        g_w = feats_w[:, self.num_query:]
        g_vis = feats_visibility[:, self.num_query:]
        g_pids = np.asarray(self.pids_local[self.num_query:])
        g_camids = np.asarray(self.camids_local[self.num_query:])

        if first_dist == 'global':
            distmat = euclidean_distance(q_gf, g_qf)
        else:
            distmat = euclidean_distance_weight(q_lf, g_lf, q_w, g_w)

        # cmc, mAP = eval_func_Recovery(distmat, q_pids, g_pids, q_camids, g_camids)

        """Evaluation with market1501 metric
               Key: for each query identity, its gallery images from the same camera view are discarded.
               """
        max_rank = 50
        num_q, num_g = distmat.shape
        # distmat g
        #    q    1 3 2 4
        #         4 1 2 3
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        #  0 2 1 3
        #  1 2 3 0
        # g_pids[indices] 相当于gallary中的pid，按相似度最前排序，行id为行人的id，(q_num,g_num)
        # q_pids[:, np.newaxis] (q_num,1)
        # matches(q_num,g_num)
        indices = torch.tensor(indices)
        g_lf_Knearst = g_lf[:, indices[:, :k_num], :]  #  (7,2210,10,1024)
        g_lf_Knearst = g_lf_Knearst.permute(1, 2, 0, 3)  # (2210, 10,7,1024)

        if recover_method == 'hard_label':
            g_w_Knearst = g_w[:, indices[:, :k_num]]
        elif recover_method == 'soft_label':
            g_w_Knearst = g_vis[:, indices[:, :k_num]]
        elif recover_method == 'mean':
            g_w_Knearst = torch.ones_like(g_w[:, indices[:, :k_num]])

        # g_w_Knearst = g_w[:, indices[:, :k_num]]  #  (7,2210,10)
        g_w_Knearst = g_w_Knearst.permute(1, 2,0).unsqueeze(dim=-1) # (2210,10,7,1)

        g_near_feat = ((g_lf_Knearst*g_w_Knearst)).sum(dim=1) / g_w_Knearst.sum(dim=1) #(2210, 7,1024)

        if recover_method == 'hard_label' or recover_method == 'mean':
            q_need_w = (1-q_w).permute(1,0) #(2210,7)
            q_recovery = q_lf.permute(1,0,2) + g_near_feat * q_need_w.unsqueeze(dim=-1)
            q_recovery = q_recovery.permute(1,0,2)

            q_recovery_w = torch.ones_like(q_w)  #默认都补全了
        else:
            q_need_w = (1-q_vis).permute(1,0) #(2210,7)
            q_recovery = q_lf.permute(1,0,2) + g_near_feat* q_need_w.unsqueeze(dim=-1)
            q_recovery = q_recovery.permute(1,0,2)
            q_recovery_w = torch.ones_like(q_vis)

        '''计算recover后的特征距离！'''
        distmat = euclidean_distance_weight(q_recovery, g_lf, q_recovery_w, g_w)
        '''打印 一下featRecover之后变化的哪些'''
        indices_feat_recovery = np.argsort(distmat, axis=1)
        indices_feat_recovery = torch.tensor(indices_feat_recovery)
        # matches_ori = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        # matches_recovery = (g_pids[indices_feat_recovery] == q_pids[:, np.newaxis]).astype(np.int32)

        import logging
        logger = logging.getLogger("transreid.train")
        pid_changes = []
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order_ori = indices[q_idx]  # select one row
            remove_ori = (g_pids[order_ori] == q_pid) & (g_camids[order_ori] == q_camid)
            keep_ori = np.invert(remove_ori)

            order_recov = indices_feat_recovery[q_idx]  # select one row
            remove_recov = (g_pids[order_recov] == q_pid) & (g_camids[order_recov] == q_camid)
            keep_recov = np.invert(remove_recov)

            indice_remove_ori = indices[q_idx][keep_ori]
            indice_remove_recov = indices_feat_recovery[q_idx][keep_recov]
            if ((indice_remove_ori[0] != indice_remove_recov[0]) and (g_pids[indice_remove_ori[0]] !=g_pids[indice_remove_recov[0]])):
                pid_changes.append(q_pid)

        # indices_feat_recovery = torch.tensor(indices_feat_recovery[:,:1])
        # indices_ori = torch.tensor(indices[:, :1])
        # change_bool = torch.tensor(g_pids[indices_ori] != g_pids[indices_feat_recovery]).squeeze()
        # change_index = torch.where(change_bool==True,1,0).nonzero()
        # # change_results_index =torch.sum((indices_feat_recovery-indices[:,:1]).bool(),dim = 1)
        # # change_index = torch.where(change_results_index>0,1,0).nonzero()
        # print(change_index)
        # print(q_pids[change_index])
        logger.info(first_dist + "-------")
        logger.info("Recovery feat imapct {} ".format(list(set(pid_changes))))
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, list(set(pid_changes))


    def compute_w_h(self):  # called after each epoch
        g_feats = torch.cat(self.g_feats, dim=0)  #[N,L]
        l_feats = torch.cat(self.l_feats, dim=1)  #[4, N, L]
        feats_w = torch.cat(self.weight, dim=1)   #[4, N]

        l_feats_w = torch.cat(self.l_feats_w, dim=1)  # [4, N, L]
        feats_w_w = torch.cat(self.weight_w, dim=1)  # [4, N]

        if self.feat_norm:
            print("The test feature is normalized")
            g_feats = torch.nn.functional.normalize(g_feats, dim=1, p=2)  # along channel

            # local_feat = [torch.nn.functional.normalize(self.l_w[i], axis=-1) for i in range(len(self.l_w))]
        # query
        # 这里可以设置只算global或者局部的特征！
        q_gf = g_feats[:self.num_query]

        q_lf = l_feats[:, :self.num_query, :]
        q_w = feats_w[:, :self.num_query]

        q_lf_w = l_feats_w[:, :self.num_query, :]
        q_w_w = feats_w_w[:, :self.num_query]

        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        g_qf = g_feats[self.num_query:]
        g_lf = l_feats[:, self.num_query:, :]
        g_w = feats_w[:, self.num_query:]


        g_lf_w = l_feats[:, self.num_query:, :]
        g_w_w = feats_w[:, self.num_query:]

        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            # print('=> Enter reranking')
            # # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            # distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
            pass
        else:
            # print('=> Computing DistMat with euclidean_distance')
            # distmat1 = euclidean_distance(q_gf, g_qf)
            # distmat2 = euclidean_distance_weight(q_lf, g_lf, q_w, g_w)
            # distmat = distmat1 + distmat2
            distmat1 = euclidean_distance_weight(q_lf, g_lf, q_w, g_w)
            distmat2 = euclidean_distance_weight(q_lf_w, g_lf_w, q_w_w, g_w_w)
            distmat = 1/2*distmat2 + 1/2*distmat1 + euclidean_distance(q_gf, g_qf)
            # distmat = distmat+ euclidean_distance(q_gf, g_qf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids

    def visualize(self, imgs_path, query_dir, gallary_dir, pids, save_dir,if_weighted=False, if_reco=False,visual_mode ="all"):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_imgs = imgs_path[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_imgs = imgs_path[self.num_query:]

        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            if if_reco:
                # print('=> Computing DistMat with euclidean_distance')
                distmat = self.compute_featRecovery(k_num=5, first_dist='local')[2]
            else:
                if if_weighted:
                    distmat = self.compute_w()[2]
                else:
                    distmat = euclidean_distance(qf, gf)


        if os.path.exists(osp.join(query_dir, q_imgs[1])):
            q_data = [[osp.join(query_dir, q_imgs[i]), q_pids[i], q_camids[i]] for i in range(len(q_imgs))]
            g_data = [[osp.join(gallary_dir, g_imgs[i]), g_pids[i], g_camids[i]] for i in range(len(g_imgs))]
        else:
            q_data = [[osp.join(query_dir,q_imgs[i][:3], q_imgs[i]), q_pids[i], q_camids[i]] for i in range(len(q_imgs))]
            g_data = [[osp.join(gallary_dir,  g_imgs[i][:3],g_imgs[i]), g_pids[i], g_camids[i]] for i in range(len(g_imgs))]
        data = [q_data, g_data]
        visualize_ranked_results(distmat,  dataset=data, pid2Visual= pids,save_dir=save_dir,mode=visual_mode)


