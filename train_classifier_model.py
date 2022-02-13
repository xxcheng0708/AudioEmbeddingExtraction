# coding:utf-8
"""
    Created by cheng star at 2022/1/15 14:17
    @email : xxcheng0708@163.com
"""
import os
import sys
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from matplotlib import pyplot as plt
import torchvision
import torch
import random
from torch import nn
import cv2
import time
import shutil
import numpy as np
from PIL import ImageOps as plops
import yaml
import argparse
from torchlibrosa.augmentation import SpecAugmentation
from utils.data_loader import AudioNpyDataset
# from models.model_resnet import AudioEmbeddingModel
from models.model_resnet_new import AudioEmbeddingModel
import warnings

warnings.filterwarnings("ignore")

os.environ["TORCH_HOME"] = "./pretrained_models"

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="./config/classifier.yaml", type=str, help="config file path")
args = parser.parse_args()
cfg_path = args.cfg

with open(cfg_path, "r", encoding="utf8") as f:
    cfg_dict = yaml.safe_load(f)
print(cfg_dict)

in_channels = cfg_dict.get("in_channels")
dataset_dir = cfg_dict.get("dataset_dir")
trainset_size = cfg_dict.get("train_size")
visible_device = cfg_dict.get("device")
batchsize = cfg_dict.get("batch_size")
train_ratio = cfg_dict.get("train_ratio")
val_ratio = cfg_dict.get("val_ratio")
test_ratio = cfg_dict.get("test_ratio")
# data augmentation
invert_ratio = cfg_dict.get("invert_ratio", 0.0)
brightness = cfg_dict.get("brightness", 0.0)
hue = cfg_dict.get("hue", 0.0)
saturation = cfg_dict.get("saturation", 0.0)
contrast = cfg_dict.get("contrast", 0.0)
left_right_flip = cfg_dict.get("left_right_flip", 0.0)
up_down_flip = cfg_dict.get("up_down_flip", 0.0)
rotate_degree = cfg_dict.get("rotate_degree", 0)
img_norm = cfg_dict.get("img_norm", False)

num_workers = cfg_dict.get("num_workers")
num_epoches = cfg_dict.get("epoch")
model_name = cfg_dict.get("model_name")
lr = cfg_dict.get("lr")
step_size = cfg_dict.get("step_size")
gamma = cfg_dict.get("gamma")
weight_decay = cfg_dict.get("weight_decay")
save_dir = cfg_dict.get("save_dir")

train_transforms_list = [
    torchvision.transforms.ToTensor()
]
val_transforms_list = [
    torchvision.transforms.ToTensor()
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
        if sample_count.get(y, 0) >= trainset_size.get(classes[y]):
            continue
        character[y].append(x)
        sample_count[y] += 1

    for i, x, in enumerate(character):
        print("{} : {}".format(classes[i], len(x)))

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

    train_dataset = AudioNpyDataset(train_inputs, train_labels, train_transforms, down_sample=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, drop_last=True,
                                  shuffle=True, num_workers=num_workers)

    val_dataset = AudioNpyDataset(val_inputs, val_labels, val_transforms, down_sample=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, drop_last=False,
                                shuffle=False, num_workers=num_workers)

    test_dataset = AudioNpyDataset(test_inputs, test_labels, val_transforms, down_sample=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, drop_last=False,
                                 shuffle=False, num_workers=num_workers)

    loader = {}
    loader["train_loader"] = train_dataloader
    loader["val_loader"] = val_dataloader
    loader["test_loader"] = test_dataloader
    return loader, classes


data_loader, classes = fetch_dataloader(dataset_dir, ratio=[train_ratio, val_ratio, test_ratio],
                                        batchsize=batchsize, num_workers=num_workers)
train_data_loader = data_loader["train_loader"]
val_data_loader = data_loader["val_loader"]
test_data_loader = data_loader["test_loader"]
print(classes)
print("train: {}, val: {}, test: {}".format(len(train_data_loader), len(val_data_loader), len(test_data_loader)))


def failure_analysis(X, y_hat, y_pred, epoch, stage, classes):
    save_path = os.path.join("./results/failure_examples", stage, str(epoch))
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    failure_index = y_pred.argmax(dim=1) != y_hat
    failure_data = X[failure_index]
    failure_label = y_pred.argmax(dim=1)[failure_index]
    true_label = y_hat[failure_index]

    failure_classes = []
    true_classes = []
    for label1, label2 in zip(failure_label.cpu().numpy(), true_label.cpu().numpy()):
        failure_classes.append(classes[label1])
        true_classes.append(classes[label2])

    for data, label1, label2 in zip(failure_data, true_classes, failure_classes):
        data = data.cpu().numpy().transpose(1, 2, 0)
        data = data * 255
        data = data.astype(np.uint8)
        timestamp = int(time.time() * 100000)
        cv2.imwrite(os.path.join(save_dir, "{}-{}-{}.jpg".format(label1, label2, timestamp)), data[:, :, ::-1])


def evaluate_accuracy_and_loss(data_iter, model, loss, epoch, error_analysis=False, stage="val"):
    acc_sum = 0.0
    loss_sum = 0.0
    n = 0

    for X, y in data_iter:
        X = X.cuda()
        y = y.cuda()
        y_pred = model(X)

        acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
        loss_sum += loss(y_pred, y).sum().item()
        n += y.shape[0]

        if error_analysis:
            failure_analysis(X, y, y_pred, epoch, stage, classes)
    return acc_sum / n, loss_sum / n


def train(model, train_iter, val_iter, loss, num_epoches, optimizer):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    lr_decay_list = []

    spec_aug = SpecAugmentation(time_drop_width=16, time_stripes_num=2,
                                freq_drop_width=8, freq_stripes_num=2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_acc = 0.0
    best_model = ""

    for epoch in range(num_epoches):
        lr_decay_list.append(optimizer.state_dict()["param_groups"][0]["lr"])
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        n = 0
        model.train()

        for batch_idx, (X, y) in enumerate(train_iter):
            X = X.cuda()
            y = y.cuda()
            if model.training:
                X = X.transpose(2, 3)
                X = spec_aug(X)
                X = X.transpose(2, 3)
            y_pred = model(X)

            l = loss(y_pred, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

            if batch_idx % 20 == 0:
                print("epoch: {}, iter: {}, lter loss: {:.4f}, iter acc: {:.4f}".format(epoch, batch_idx, l.item(), (
                        y_pred.argmax(dim=1) == y).float().mean().item()))
        lr_scheduler.step()
        model.eval()

        t_acc, t_loss = evaluate_accuracy_and_loss(train_iter, model, loss, epoch, error_analysis=True, stage="train")
        v_acc, v_loss = evaluate_accuracy_and_loss(val_iter, model, loss, epoch, error_analysis=True, stage="val")
        train_acc.append(train_acc_sum / n)
        train_loss.append(train_loss_sum / n)
        # train_acc.append(t_acc)
        # train_loss.append(t_loss)
        val_acc.append(v_acc)
        val_loss.append(v_loss)

        print("epoch: {}, train acc: {:.4f}, train loss: {:.4f}, val acc: {:.4f}, val loss: {:.4f}".format(
            epoch, train_acc[-1], train_loss[-1], val_acc[-1], val_loss[-1]))
        if v_acc > best_acc:
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)
            best_acc = v_acc
            best_model = os.path.join(save_dir, "model-{}-{}-{}.pth".format(model_name, epoch, best_acc))
            torch.save(model.module.state_dict(), best_model)
    return train_acc, train_loss, val_acc, val_loss, lr_decay_list, best_model, best_acc


# model, optimizer = build_model(model_name, out_features=len(classes), pretrained=True)
model = AudioEmbeddingModel(input_dimension=64, out_dimension=len(classes), in_channels=in_channels,
                            model_name="resnet18", pretrained=True)
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
    weight_decay=weight_decay
)
model = torch.nn.DataParallel(model, device_ids=visible_device).cuda()
print(model)
print(model.device_ids)

train_iter, val_iter, test_iter = train_data_loader, val_data_loader, test_data_loader
loss = torch.nn.CrossEntropyLoss()

train_acc, train_loss, val_acc, val_loss, lr_decay_list, best_model, best_acc = train(
    model, train_iter, val_iter, loss, num_epoches, optimizer)

print("best model: {}, best accuracy: {}".format(best_model, best_acc))

# model, _ = build_model(model_name, out_features=len(classes), pretrained=False)
model = AudioEmbeddingModel(input_dimension=64, out_dimension=len(classes), in_channels=in_channels,
                            model_name="resnet18", pretrained=False)
model.load_state_dict(torch.load(best_model))
model = model.cuda()
model.eval()
test_acc, test_loss = evaluate_accuracy_and_loss(test_data_loader, model, loss,
                                                 epoch="test", error_analysis=True, stage="val")
print("test accuracy: {}, test loss: {}".format(test_acc, test_loss))

fig, axes = plt.subplots(1, 3)
axes[0].plot(list(range(1, num_epoches + 1)), train_loss, color="r", label="train loss")
axes[0].plot(list(range(1, num_epoches + 1)), val_loss, color="b", label="validate loss")
axes[0].legend()
axes[0].set_title("Loss")

axes[1].plot(list(range(1, num_epoches + 1)), train_acc, color="r", label="train acc")
axes[1].plot(list(range(1, num_epoches + 1)), val_acc, color="b", label="validate acc")
axes[1].legend()
axes[1].set_title("Accuracy")

axes[2].plot(list(range(1, num_epoches + 1)), lr_decay_list, color="r", label="lr")
axes[2].legend()
axes[2].set_title("Learning Rate")
plt.show()
