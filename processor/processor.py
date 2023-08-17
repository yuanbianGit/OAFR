import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import datetime
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
import torch.distributed as dist
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.vehicleid import VehicleID
from datasets.veri import VeRi
from datasets.partial_reid import Paritial_REID
from datasets.occlusion_reid import Occluded_REID

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'occ_reid': Occluded_REID,
    'partial_reid': Paritial_REID,
}

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    loss_meter1 = AverageMeter()
    loss_meter2 = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        loss_meter2.reset()
        loss_meter1.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = img.to(device)
            target = vid.to(device)

            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):

                if cfg.MODEL.OCC_AUG:
                    target = torch.cat([target, target], dim=0)
                    target_cam = torch.cat([target_cam, target_cam], dim=0)
                if cfg.MODEL.ARC == 'OAFR':
                    score, feat, occ_feat, occ_lable_gt_2d, occ_lable_gt, occ_pre_label, local_feat, rand_index, from_copy_tensor_label = model(
                        img, cfg.MODEL.OCC_AUG, cfg.SOLVER.PERSON_OCC_PRO, target, cam_label=target_cam, view_label=target_view)

                    loss_1 = loss_fn(score, feat, target, target_cam)

                    loss_2 = torch.nn.functional.cross_entropy(occ_pre_label, occ_lable_gt)
                    loss = loss_1 + loss_2
                else:
                    score, feat = model(img, cfg.MODEL.OCC_AUG, cfg.SOLVER.PERSON_OCC_PRO,  target, cam_label=target_cam, view_label=target_view)
                    loss = loss_fn(score, feat, target, target_cam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)


            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            if cfg.MODEL.ARC == 'OAFR':
                loss_meter1.update(loss_1,img.shape[0])
                loss_meter2.update(loss_2,img.shape[0])

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f},Loss_pid: {:.3f},Loss_occPredict: {:.3f} Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, loss_meter1.avg,loss_meter2.avg,scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME +datetime.datetime.now().strftime('%m%d%H%M') + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME +datetime.datetime.now().strftime('%m%d%H%M') + '_{}.pth'.format(epoch)))


        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)

                            if cfg.MODEL.ARC == 'OAFR':
                                globla_feats, local_feat_h, wei_h, co_h_pro = model(img, cfg.SOLVER.PERSON_OCC_PRO, cam_label=camids, view_label=target_view)
                                evaluator.update((globla_feats, vid, camid))
                                evaluator.update_w((globla_feats, local_feat_h, wei_h, local_feat_h, co_h_pro, vid, camid))
                            else:
                                feat = model(img, cam_label=camids, view_label=target_view)
                                evaluator.update((feat, vid, camid))
                    if cfg.MODEL.ARC == 'OAFR':
                        cmc, mAP, _, _, _, _, _ = evaluator.compute()
                        cmc_local, mAP_local, _, _, _ = evaluator.compute_w()
                        cmc_local_recovery, mAP_local_recovery, _, _, _ , _ = evaluator.compute_featRecovery(k_num=cfg.SOLVER.NEAREST_K)

                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                        logger.info("local_mAP: {:.1%}".format(mAP_local))
                        for r in [1, 5, 10]:
                            logger.info("local_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_local[r - 1]))
                        logger.info("local_mAP: {:.1%}".format(mAP_local_recovery))
                        for r in [1, 5, 10]:
                            logger.info("local_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_local_recovery[r - 1]))
                    else:
                        cmc, mAP, _, _, _, _, _ = evaluator.compute()
                        logger.info("Validation Results - Epoch: {}".format(epoch))
                        logger.info("mAP: {:.1%}".format(mAP))
                        for r in [1, 5, 10]:
                            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        if cfg.MODEL.ARC == 'OAFR':
                            globla_feats, local_feat_h, wei_h, co_h_pro = model(img, cfg.SOLVER.PERSON_OCC_PRO,
                                                                                cam_label=camids,
                                                                                view_label=target_view)
                            evaluator.update((globla_feats, vid, camid))
                            evaluator.update_w((globla_feats, local_feat_h, wei_h, local_feat_h, co_h_pro, vid, camid))
                        else:
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                if cfg.MODEL.ARC == 'OAFR':
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    cmc_local, mAP_local, _, _, _ = evaluator.compute_w()
                    cmc_local_recovery, mAP_local_recovery, _, _, _ ,_= evaluator.compute_featRecovery(
                        k_num=cfg.SOLVER.NEAREST_K,first_dist='local',recover_method = cfg.SOLVER.RECOVER_METHOD)
                    cmc_global_recovery, mAP_global_recovery, _, _, _,_ = evaluator.compute_featRecovery(
                        k_num=cfg.SOLVER.NEAREST_K, first_dist='global',recover_method = cfg.SOLVER.RECOVER_METHOD)

                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


                    logger.info("local_mAP: {:.1%}".format(mAP_local))
                    for r in [1, 5, 10]:
                        logger.info("local_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_local[r - 1]))

                    logger.info("local_recovery_mAP: {:.1%}".format(mAP_local_recovery))
                    for r in [1, 5, 10]:
                        logger.info("local_recovery_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_local_recovery[r - 1]))

                    logger.info("global_recovery_mAP: {:.1%}".format(mAP_global_recovery))
                    for r in [1, 5, 10]:
                        logger.info(
                            "local_recovery_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_global_recovery[r - 1]))
                else:
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    pid_changes = []
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            img_path_list.extend(imgpath)

            if cfg.MODEL.ARC == 'OAFR':
                globla_feats, local_feat_h, wei_h, co_h_pro = model(img, cfg.SOLVER.PERSON_OCC_PRO,
                                                                    cam_label=camids,
                                                                    view_label=target_view)
                evaluator.update((globla_feats, pid, camid))
                evaluator.update_w((globla_feats, local_feat_h, wei_h, local_feat_h, co_h_pro, pid, camid))
            else:
                feat = model(img, cam_label=camids, view_label=target_view)
                evaluator.update((feat, pid, camid))


    if cfg.MODEL.ARC == 'OAFR':
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        cmc_local, mAP_local, _, _, _ = evaluator.compute_w()
        cmc_local_recovery, mAP_local_recovery, _, _, _ ,pid_changes = evaluator.compute_featRecovery(
            k_num=cfg.SOLVER.NEAREST_K, first_dist='local',recover_method = cfg.SOLVER.RECOVER_METHOD)
        cmc_global_recovery, mAP_global_recovery, _, _, _ ,_= evaluator.compute_featRecovery(
            k_num=cfg.SOLVER.NEAREST_K, first_dist='global',recover_method = cfg.SOLVER.RECOVER_METHOD)

        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        logger.info("local_mAP: {:.1%}".format(mAP_local))
        for r in [1, 5, 10]:
            logger.info("local_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_local[r - 1]))

        logger.info("local_recovery_mAP: {:.1%}".format(mAP_local_recovery))
        for r in [1, 5, 10]:
            logger.info("local_recovery_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_local_recovery[r - 1]))

        logger.info("global_recovery_mAP: {:.1%}".format(mAP_global_recovery))
        for r in [1, 5, 10]:
            logger.info(
                "local_recovery_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_global_recovery[r - 1]))
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    if cfg.TEST.VISUALIZE ==True:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
        if cfg.TEST.VISUAL_FEAT_RECOVERY and cfg.TEST.PID_VISUAL==[]:
            evaluator.visualize(img_path_list, dataset.query_dir, dataset.gallery_dir, pids=pid_changes, save_dir=cfg.TEST.VISUAL_DIR, if_weighted = cfg.TEST.LOCAL,if_reco=cfg.TEST.VISUAL_FEAT_RECOVERY,visual_mode = cfg.TEST.VISUAL_MODE)
        else:
            evaluator.visualize(img_path_list, dataset.query_dir, dataset.gallery_dir, pids=cfg.TEST.PID_VISUAL, save_dir=cfg.TEST.VISUAL_DIR, if_weighted = cfg.TEST.LOCAL,if_reco=cfg.TEST.VISUAL_FEAT_RECOVERY,visual_mode = cfg.TEST.VISUAL_MODE)

    return cmc[0], cmc[4]
