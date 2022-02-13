# coding:utf-8
"""
    Created by cheng star at 2022/1/22 20:23
    @email : xxcheng0708@163.com
"""
from pytorch_metric_learning import losses, distances, miners
import torch


def get_loss(loss_name, xbm_enable, lr, weight_decay, classes_num=0, out_dimension=128, xbm_size=10240):
    distance = None
    loss_miner = None
    loss_func = None
    xbm_loss_func = None
    loss_optimizer = None

    ### Paid Based Loss
    if loss_name == "TripletMarginLoss":
        # TripletMarginLoss
        distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        loss_miner = miners.TripletMarginMiner(margin=0.5, distance=distance, type_of_triplets="semihard")
        loss_func = losses.TripletMarginLoss(margin=0.5, distance=distance, swap=True)
        xbm_loss_func = None
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    if loss_name == "MultiSimilarityLoss":
        # MultiSimilarityLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, distance=distance)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    if loss_name == "ContrastiveLoss":
        # ContrastiveLoss
        distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        loss_miner = miners.PairMarginMiner(pos_margin=0.1, neg_margin=1.0)
        loss_func = losses.ContrastiveLoss(distance=distance, pos_margin=0.1, neg_margin=1.0)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    if loss_name == "CircleLoss":
        # Circle Loss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.CircleLoss(m=0.25, gamma=400, distance=distance)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    if loss_name == "SupervisedContrastiveLoss":
        # SupervisedContrastiveLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.SupConLoss(temperature=0.1, distance=distance)
        if xbm_enable:
            xbm_loss_func = get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner)

    ### CrossEntropy Loss(requires an loss optimizer)
    if loss_name == "ArcFaceLoss":
        # ArcFaceLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.ArcFaceLoss(num_classes=classes_num, embedding_size=out_dimension, margin=0.5, scale=8)

    if loss_name == "ProxyAnchorLoss":
        # ProxyAnchorLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.ProxyAnchorLoss(num_classes=classes_num, embedding_size=out_dimension, margin=0.1, alpha=32)

    if loss_name == "SoftTripletLoss":
        # SoftTripletLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.SoftTripleLoss(num_classes=classes_num, embedding_size=out_dimension)

    if loss_name == "LargeMarginSoftmaxLoss":
        # LargeMarginSoftmaxLoss
        distance = distances.CosineSimilarity()
        loss_miner = miners.MultiSimilarityMiner()
        loss_func = losses.LargeMarginSoftmaxLoss(num_classes=classes_num, embedding_size=out_dimension)

    if classes_num != 0:
        loss_optimizer = torch.optim.SGD(
            loss_func.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    return loss_func, xbm_loss_func, loss_miner, loss_optimizer


def get_xbm_loss_func(loss_func, xbm_size, out_dimension, loss_miner=None):
    xbm_loss_func = losses.CrossBatchMemory(loss=loss_func, embedding_size=out_dimension,
                                            memory_size=xbm_size,
                                            miner=loss_miner)
    return xbm_loss_func
