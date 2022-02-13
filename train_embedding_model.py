# coding:utf-8
"""
    Created by cheng star at 2022/1/15 14:17
    @email : xxcheng0708@163.com
"""
import os

os.environ["TORCH_HOME"] = "./data/pretrained_models"
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import random
from torch import nn
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pytorch_metric_learning import samplers, losses, miners, testers, distances, reducers
from utils.transforms import MinMaxNormalize, MeanStdNormalize, ZeroMeanNormalize
# from models.model_resnet import AudioEmbeddingModel
from models.model_resnet_new import AudioEmbeddingModel
# from models.model_simple import AudioEmbeddingModel
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from utils.data_prefetcher import DataPrefetcher
from utils.mixup import mixup_data
from utils.losses import get_loss, get_xbm_loss_func
from utils.data_loader import AudioNpyDataset
from preprocessing.data_normalize import get_dataset_mean_and_std
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./config/embedding.yaml", type=str, help="config file path")
args = parser.parse_args()
cfg_path = args.cfg

with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
print(cfg_dict)

data_type = cfg_dict.get("data_type")
in_channels = cfg_dict.get("in_channels")
train_dataset_dir = cfg_dict.get("train_dataset_dir")
val_dataset_dir = cfg_dict.get("val_dataset_dir")
validate_loss = cfg_dict.get("validate_loss")
xbm_enable = cfg_dict.get("xbm_enable")
mixup_enable = cfg_dict.get("mixup_enable")
mean_std_enable = cfg_dict.get("mean_std_enable")
xbm_size = cfg_dict.get("xbm_size")
xbm_start_iteration = cfg_dict.get("xbm_start_iteration")
xbm_weight = cfg_dict.get("xbm_weight")
xbm_mixup_alternate = cfg_dict.get("xbm_mixup_alternate")
out_dimension = cfg_dict.get("out_dimension")
input_dimension = cfg_dict.get("input_dimension")
visible_device = cfg_dict.get("device")
batchsize = cfg_dict.get("batch_size")
train_ratio = cfg_dict.get("train_ratio")
val_ratio = cfg_dict.get("val_ratio")
test_ratio = cfg_dict.get("test_ratio")
invert_ratio = cfg_dict.get("invert_ratio")
num_workers = cfg_dict.get("num_workers")
num_epoches = cfg_dict.get("epoch")
lr = cfg_dict.get("lr")
step_size = cfg_dict.get("step_size")
gamma = cfg_dict.get("gamma")
weight_decay = cfg_dict.get("weight_decay")
left_right_flip = cfg_dict.get("left_right_flip")
up_down_flip = cfg_dict.get("up_down_flip")
save_dir = cfg_dict.get("save_dir")
pretrained_model_path = cfg_dict.get("pretrained_models")

train_transforms_list = [
    torchvision.transforms.ToTensor(),
    ZeroMeanNormalize()
]
val_transforms_list = [
    torchvision.transforms.ToTensor(),
    ZeroMeanNormalize()
]

train_transforms = torchvision.transforms.Compose(train_transforms_list)
val_transforms = torchvision.transforms.Compose(val_transforms_list)


def fetch_dataloader(dataset_dir, ratio, batchsize=512, num_workers=8, seed=100,
                     train_transforms=train_transforms, val_transforms=val_transforms):
    random.seed(seed)

    dataset = torchvision.datasets.DatasetFolder(dataset_dir, loader=np.load, extensions=("npy",))
    classes = dataset.classes
    character = [[] for _ in range(len(classes))]
    random.shuffle(dataset.samples)
    sample_count = {}

    for x, y in dataset.samples:
        if y not in sample_count:
            sample_count[y] = 0
        character[y].append(x)
        sample_count[y] += 1

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
        for x in data[num_val_index:]:
            test_inputs.append(str(x))
            test_labels.append(i)

    print("train_inputs: {}, train_labels: {}".format(len(train_inputs), len(train_labels)))
    print("val_inputs: {}, val_labels: {}".format(len(val_inputs), len(val_labels)))
    print("test_inputs: {}, test_labels: {}".format(len(test_inputs), len(test_labels)))

    train_dataset = AudioNpyDataset(train_inputs, train_labels, train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, drop_last=True,
                                  shuffle=False, num_workers=num_workers,
                                  sampler=samplers.MPerClassSampler(train_labels, m=8, batch_size=batchsize,
                                                                    length_before_new_iter=len(train_labels)),
                                  pin_memory=True)
    if mean_std_enable:
        mean, std = get_dataset_mean_and_std(train_dataset, batchsize)
        print("train dataset, mean: {}, std: {}".format(mean, std))

    val_dataset = None
    val_dataloader = None
    test_dataset = None
    test_dataloader = None

    if len(val_inputs):
        val_dataset = AudioNpyDataset(val_inputs, val_labels, val_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batchsize, drop_last=False,
                                    shuffle=False, num_workers=num_workers,
                                    sampler=samplers.MPerClassSampler(val_labels, m=8, batch_size=batchsize,
                                                                      length_before_new_iter=len(val_labels)),
                                    pin_memory=True)

    if len(test_inputs) >= batchsize:
        test_dataset = AudioNpyDataset(test_inputs, test_labels, val_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, drop_last=False,
                                     shuffle=False, num_workers=num_workers,
                                     sampler=samplers.MPerClassSampler(test_labels, m=8, batch_size=batchsize,
                                                                       length_before_new_iter=len(test_labels)),
                                     pin_memory=True)

    loader = {}
    loader["train_set"] = train_dataset
    loader["val_set"] = val_dataset
    loader["test_set"] = test_dataset
    loader["train_loader"] = train_dataloader
    loader["val_loader"] = val_dataloader
    loader["test_loader"] = test_dataloader
    return loader, classes


# 加载训练集
# 度量学习模型的验证集和测试集要独立于训练集，所以train_dataset_dir数据全部用来训练
data_loader, classes = fetch_dataloader(train_dataset_dir, ratio=[1.0, 0.0, 0.0],
                                        batchsize=batchsize, num_workers=num_workers,
                                        train_transforms=train_transforms, val_transforms=val_transforms)
train_data_loader = data_loader["train_loader"]

# load validate dataset and test dataset
extra_data_loader, _ = fetch_dataloader(val_dataset_dir, ratio=[0.7, 0.3, 0.0],
                                        batchsize=batchsize, num_workers=num_workers,
                                        train_transforms=val_transforms, val_transforms=val_transforms)
extra_train_data_loader = extra_data_loader["train_loader"]
extra_val_data_loader = extra_data_loader["val_loader"]
extra_train_dataset = extra_data_loader["train_set"]
extra_val_dataset = extra_data_loader["val_set"]
print("train dataloader: {}, val dataloader: {}, test dataloader: {}".format(
    len(train_data_loader), len(extra_train_data_loader), len(extra_val_data_loader)))


def get_all_embeddings(dataset, model):
    tester = testers.GlobalEmbeddingSpaceTester()
    return tester.get_all_embeddings(dataset, model)


def test(std_dataset, query_dataset, model, accuracy_calculator, knn_model=None):
    model.eval()
    std_embeddings, std_labels = get_all_embeddings(std_dataset, model)
    query_embeddings, query_labels = get_all_embeddings(query_dataset, model)

    std_labels = std_labels.squeeze(1)
    query_labels = query_labels.squeeze(1)

    std_embeddings = std_embeddings.cpu()
    std_labels = std_labels.cpu()
    query_embeddings = query_embeddings.cpu()
    query_labels = query_labels.cpu()

    accuracies = accuracy_calculator.get_accuracy(query_embeddings, std_embeddings,
                                                  query_labels, std_labels, False)
    knn_score = 0.0

    if knn_model is not None:
        scores = cross_val_score(knn_model, query_embeddings, query_labels, cv=5, scoring="accuracy")
        knn_score = np.mean(scores)
        print("KNN accuracy: {}".format(knn_score))
    accuracies["knn_score"] = knn_score
    print(accuracies)
    return accuracies


def evaluate(data_iter, net, loss_func, mining_func):
    """
    evaluate validate loss
    :param data_iter:
    :param net:
    :param loss_func:
    :param mining_func:
    :return:
    """
    net.eval()
    loss_sum = 0.0
    n = 0
    for batch_idx, (X, y) in enumerate(data_iter):
        X = X.cuda()
        y = y.cuda()
        y_pred = net(X)
        if mining_func:
            indices_tuple = mining_func(y_pred, y)
            loss = loss_func(y_pred, y, indices_tuple)
        else:
            loss = loss_func(y_pred, y)

        loss_sum += loss.item()
        n += y.shape[0]

        if batch_idx % 20 == 0:
            print("validation, Iteration {}: Loss = {}".format(batch_idx, loss))
    return loss_sum / n


def train(model, train_iter, train_set, val_iter, val_set, loss_func_list, num_epoches, accuracy_calculator,
          optimizer=None, loss_optimizer=None, mining_func=None, writer=None, knn_model=None, mixup_enable=False):
    lr_decay_list = []
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    print("xbm start iteration: {}, xbm size: {}, mixup enable: {}".format(xbm_start_iteration, xbm_size, mixup_enable))
    best_acc = 0.0
    best_loss = 1e-10
    loss_func = loss_func_list[0]
    for epoch in range(num_epoches):
        model.train()
        lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
        train_loss_sum = 0.0
        n = 0

        prefetcher = DataPrefetcher(train_iter)
        X, y = prefetcher.next()
        batch_idx = 0
        while X is not None:
            batch_idx += 1

            # use xbm loss function after xbm_start_iteration
            # loss_func = loss_func_list[0]
            # if xbm_enable and epoch * len(train_iter) + batch_idx >= xbm_start_iteration:
            #     loss_func = loss_func_list[1]
            #     mixup_enable = False

            if xbm_enable and epoch * len(train_iter) + batch_idx >= xbm_start_iteration:
                if xbm_mixup_alternate and mixup_enable is False and \
                        (epoch * len(train_iter) + batch_idx - xbm_start_iteration) // xbm_mixup_alternate % 2 == 1:
                    # switch to mixup
                    loss_func = loss_func_list[0]
                    mixup_enable = True
                    print("switch to mixup, iterations: {}, mixup enable: {}".format(epoch * len(train_iter) + batch_idx,
                                                                                     mixup_enable))

                elif xbm_mixup_alternate and mixup_enable is True and \
                        (epoch * len(train_iter) + batch_idx - xbm_start_iteration) // xbm_mixup_alternate % 2 == 0:
                    # switch to xbm
                    loss_func = get_xbm_loss_func(loss_func_list[0], xbm_size, out_dimension, mining_func)
                    mixup_enable = False
                    print("switch to xbm, iterations: {}, mixup enable: {}".format(epoch * len(train_iter) + batch_idx,
                                                                                   mixup_enable))

            X = X.cuda()
            y = y.cuda()

            if mixup_enable:
                X, y_a, y_b, lam = mixup_data(X, y, 1.0, True)
                X, y_a, y_b = map(torch.autograd.Variable, (X, y_a, y_b))
                X = X.cuda()
                y_a = y_a.cuda()
                y_b = y_b.cuda()

                y_pred = model(X)
                loss = lam * loss_func(y_pred, y_a) + (1 - lam) * loss_func(y_pred, y_b)
            else:
                y_pred = model(X)
                if mining_func:
                    indices_tuple = mining_func(y_pred, y)
                    loss = loss_func(y_pred, y, indices_tuple)
                else:
                    loss = loss_func(y_pred, y)

            loss = torch.mean(loss)

            optimizer.zero_grad()
            if loss_optimizer:
                loss_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss_optimizer:
                loss_optimizer.step()

            train_loss_sum += loss.item()
            n += y.shape[0]

            if batch_idx % 20 == 0:
                print("Epoch {} Iteration {}, Loss = {}".format(epoch, batch_idx, loss))
                writer.add_scalar("loss/train", loss.item(), epoch * len(train_iter) + batch_idx)

            if batch_idx % 2000 == 0:
                torch.cuda.empty_cache()

                with torch.no_grad():
                    if validate_loss:
                        val_loss = evaluate(val_iter, model, loss_func_list[0], mining_func)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_model_name = os.path.join(save_dir, "model-loss-%03d-%03d-%.06f.pth" % (
                                epoch + 1, batch_idx, best_loss))
                            torch.save(model.module.state_dict(), best_model_name)
                        print("Epoch {} Iteration {}, Val Loss ={}".format(epoch, batch_idx, val_loss))
                        writer.add_scalar("loss/val", val_loss, epoch * len(train_iter) + batch_idx)

                    acc = test(train_set, val_set, model, accuracy_calculator, knn_model)
                    precision_at_1 = acc["precision_at_1"]
                    r_precision = acc["r_precision"]
                    r_map = acc["mean_average_precision_at_r"]
                    knn_score = acc["knn_score"]
                    print(acc)

                    writer.add_scalar("accuracy/precision@1", precision_at_1, epoch * len(train_iter) + batch_idx)
                    writer.add_scalar("accuracy/r_precision", r_precision, epoch * len(train_iter) + batch_idx)
                    writer.add_scalar("accuracy/mean_average_precision_at_r", r_map,
                                      epoch * len(train_iter) + batch_idx)
                    writer.add_scalar("accuracy/knn_score", knn_score, epoch * len(train_iter) + batch_idx)

                    if r_map > best_acc:
                        best_acc = r_map
                        best_model_name = os.path.join(save_dir, "model-acc-%03d-%03d-%.04f-%.04f-%.04f.pth" % (
                            epoch + 1, batch_idx, best_acc, precision_at_1, r_precision))
                        torch.save(model.module.state_dict(), best_model_name)
            model.train()
            X, y = prefetcher.next()

        print("Epoch {}, Train Loss = {}".format(epoch, train_loss_sum / n))
        writer.add_scalar("loss/train", train_loss_sum / n, epoch)
        writer.add_scalar("learning rate", lr_decay_list[-1], epoch)
        lr_scheduler.step()


model = AudioEmbeddingModel(input_dimension=input_dimension, out_dimension=out_dimension, model_name="resnet18",
                            pretrained=True)
train_parameters = list(map(id, model.model.fc.parameters()))
if in_channels != 3:
    train_parameters.extend(list(map(id, model.model.conv1.parameters())))
pretrained_parameters = filter(lambda p: id(p) not in train_parameters, model.model.parameters())
train_parameters = filter(lambda p: id(p) in train_parameters, model.model.parameters())

optimizer = torch.optim.SGD(
    [
        {"params": train_parameters, "lr": lr},
        {"params": pretrained_parameters, "lr": lr / 10}
    ],
    lr=lr,
    weight_decay=weight_decay,
    momentum=0.9
)
model = torch.nn.DataParallel(model, device_ids=visible_device).cuda()
print(model)
print(model.device_ids)

loss_func, xbm_loss_func, loss_miner, loss_optimizer = get_loss("CircleLoss", xbm_enable, lr, weight_decay,
                                                                classes_num=0, out_dimension=out_dimension,
                                                                xbm_size=xbm_size)

accuracy_calculator = AccuracyCalculator(
    include=(
        "precision_at_1",
        "r_precision",
        "mean_average_precision_at_r"
    ),
    k="max_bin_count"
)

knn_model = KNeighborsClassifier(n_neighbors=5, p=2, n_jobs=-1)
train_iter, val_iter, test_iter = train_data_loader, extra_train_data_loader, extra_val_data_loader

writer = SummaryWriter(save_dir)
train(model, train_iter, extra_train_dataset, val_iter, extra_val_dataset, [loss_func, xbm_loss_func], num_epoches,
      accuracy_calculator, optimizer=optimizer, loss_optimizer=loss_optimizer,
      mining_func=loss_miner, writer=writer, knn_model=None)
writer.close()
