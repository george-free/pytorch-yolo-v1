#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data import DataLoader
import os
from glob import glob
import random
import cv2
import numpy as np
from torchsummary import summary
from torch.optim import lr_scheduler

class Yolov1(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=3):
        super(Yolov1, self).__init__()
        self._num_classes = num_classes
        self._in_channels = in_channels

        # create vgg16 layers
        self._back_bone = self.vgg16(self._in_channels)
        
        # create classifiers
        self._classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 7 * 7 * (10 + self._num_classes))
        )

    def vgg16(self, in_channels, batch_norm=True):
        layers = []
        conv1_config = {
            "n_layer": 3,
            "n_feature_map": 64,
            "stride": 1,
            "kernel_size": 3,
            "padding": 1,
        }
        conv2_config = {
            "n_layer": 2,
            "n_feature_map": 128,
            "stride": 1,
            "kernel_size": 3,
            "padding": 1,
        }
        conv3_config = {
            "n_layer": 3,
            "n_feature_map": 256,
            "stride": 1,
            "kernel_size": 3,
            "padding": 1,
        }
        conv4_config = {
            "n_layer": 3,
            "n_feature_map": 512,
            "stride": 1,
            "kernel_size": 3,
            "padding": 1,
        }
        conv5_config = {
            "n_layer": 3,
            "n_feature_map": 512,
            "stride": 1,
            "kernel_size": 3,
            "padding": 1,
        }
        configs = [conv1_config, conv2_config, conv3_config,
                   conv4_config, conv5_config]
        for conf in configs:
            for step in range(0, conf.get("n_layer")):
                conv2d = nn.Conv2d(
                    in_channels, conf["n_feature_map"],
                    kernel_size=conf["kernel_size"],
                    stride=conf["stride"],
                    padding=conf["padding"])
                
                if batch_norm is True:
                    layers += [conv2d, nn.BatchNorm2d(conf["n_feature_map"]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = conf["n_feature_map"]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self._back_bone(x)
        # bachsize * (512 * 7 * 7)
        # print("backbone output shape: {}".format(x.size()))
        x = x.view(x.size(0), -1)
        x = self._classifier(x)
        x = nn.Sigmoid()(x)
        x = x.view(-1, 7, 7, 10 + self._num_classes)
        
        return x
        
class DataSet(torchDataset):
    def __init__(self, 
                 data_dir,
                 mean_data=None,
                 std_data=None,
                 img_size=[224, 224],
                 step="train") -> None:
        if not os.path.exists(data_dir):
            print("error - {} does not exist".format(data_dir))
        self.__input_dim = img_size  # height, width
        self._class_ids = {
            "face_mask": 1,
            "no_face_mask": 2,
            "half_mask": 3
        }
        self._num_classes = len(self._class_ids.keys())
        self._image_ids = []
        
        image_id = 1
        for image_path in glob(data_dir + "/*.jpg"):
            # single_data_name = image_path.split("/")[-1][0:-4]
            txt_path = image_path[0:-4] + ".txt"
            self._image_ids.append([image_id, image_path, txt_path])
            image_id += 1
            
        # shuffle data
        random.shuffle(self._image_ids)
        if mean_data is None or std_data is None:
            self._image_mean, self._image_std = self.cal_mean_std(self._image_ids)
        else:
            self._image_mean = mean_data
            self._image_std = std_data
        self._num_images = len(self._image_ids)
        
        # label format
        self._label_delimiter = " "
        
    @property
    def input_dim(self):
        return self.__input_dim
            
    def cal_mean_std(self, data_list):
        means = []
        variances = []
        for data in data_list:
            _, image_path, _ = data
            # print("img path: {}".format(image_path))
            img = cv2.imread(image_path)
            img = cv2.resize(
                img,
                (self.__input_dim[1], self.__input_dim[0]),
                interpolation=cv2.INTER_LINEAR)
            # img = img.transpose((2, 0, 1))
            # print(img.shape)
            means.append(np.mean(img, axis=(0, 1)))
        means = np.array(means, dtype=np.float32)
        # print("means shape: {}".format(means.shape))
        mean_bgr = np.mean(means, axis=0)
        for data in data_list:
            _, image_path, _ = data
            img = cv2.imread(image_path)
            img = cv2.resize(
                img,
                (self.__input_dim[1], self.__input_dim[0]),
                interpolation=cv2.INTER_LINEAR)
            # img = img.transpose((2, 0, 1))
            img_var = np.mean((img - mean_bgr) ** 2, axis=(0, 1))
            variances.append(img_var)

        std_bgr = np.sqrt(np.mean(variances, axis=0))
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
        std_rgb = np.array([std_bgr[2], std_bgr[1], std_bgr[0]])
        
        print("mean: {}".format(mean_rgb))
        print("std: {}".format(std_rgb))
        return mean_rgb, std_rgb
    
    @classmethod
    def preproc_image(cls,
                      img,
                      input_dim,
                      mean,
                      std,
                      swap=(2, 0, 1),
                      padding:int=114):
        assert len(img.shape) == 3
        padded_resized_image = cls.resize_image(img, input_dim, padding)
        
        # normalzied
        for channel in range(0, 3):
            padded_resized_image[channel] = (padded_resized_image[channel] - mean[channel]) / std[channel]
        padded_resized_image = padded_resized_image.transpose(swap)
        
        return padded_resized_image
    
    @classmethod
    def preproc_labels(cls, boxes, labels, image_size, num_classes, grid_num=7):
        '''
            boxes tensor
            labels tensor
            return 7 x 7 x M  tensors
        '''
        target = torch.zeros((grid_num, grid_num, 10 + num_classes))
        if boxes.size(0) > 0:
            cell_size = image_size / grid_num
            boxes_wh = boxes[:, 2:4]
            # print("boxes wh: {}".format(boxes_wh))
            if np.any(boxes_wh.numpy() < 1):
                boxes_wh = torch.tensor([image_size[1], image_size[0]]).expand_as(boxes_wh) * boxes_wh
            # boxes_cxcy = (boxes[:, 0:2] + boxes[:, 2:]) * 0.5  # x1 y1, x2 ,y2
            boxes_cxcy = boxes[:, 0:2]
            if np.any(boxes_cxcy.numpy() < 1):
                boxes_cxcy = torch.tensor([image_size[1], image_size[0]]).expand_as(boxes_wh) * boxes_cxcy
        
            num_boxes = boxes.shape[0]
            for index in range(0, num_boxes):
                ij = (boxes_cxcy[index] / cell_size).ceil() - 1
                xy = ij * cell_size
                delta_xy = (boxes_cxcy[index] - xy) / cell_size
                # print(f"target: {cell_size}, {ij}, {xy}, {delta_xy}, {boxes_wh}")
                target[int(ij[1]), int(ij[0]), 0:2] = delta_xy
                target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
                target[int(ij[1]), int(ij[0]), 2:4] = boxes_wh[index]  # absolute size of the box
                target[int(ij[1]), int(ij[0]), 7:9] = boxes_wh[index]
                target[int(ij[1]), int(ij[0]), 4] = 1
                target[int(ij[1]), int(ij[0]), 9] = 1
                target[int(ij[1]), int(ij[0]), int(labels[index] + 9)] = 1

        return target
    
    def __getitem__(self, index):
        image_id, image_path, txt_path = self._image_ids[index]
        
        # step 1. deal with image data
        img = cv2.imread(image_path)
        assert img is not None, f"file named {image_path} not found"
        img_szie = img.shape[0:2]
        img = self.preproc_image(
            img, self.__input_dim, self._image_mean, self._image_std)
        
        boxes = []
        labels = []
        with open(txt_path, "r") as f:
            for obj in f.readlines():
                obj = obj.strip("\n")
                # print("obj: {}".format(obj))
                if len(obj) < 1:
                    continue
                cls_id = int(obj.split(self._label_delimiter)[0])
                box = obj.split(self._label_delimiter)[1:]
                box = list(map(float, box))
                # print(cls_id, box)
                boxes.append(box)
                labels.append(cls_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        image_size = torch.tensor(self.__input_dim)
        targets = self.preproc_labels(boxes, labels, image_size, self._num_classes)
        return img, targets

    def __len__(self):
        return len(self._image_ids)

    @classmethod
    def resize_image(cls, img, input_dim, padding=114):
        padded_image = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * padding
        ratio = min(input_dim[0] / img.shape[0],
                    input_dim[1] / img.shape[1])
        try:
            resized_img = cv2.resize(img, 
                                    (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                                    interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            padded_image[0:resized_img.shape[0], 0:resized_img.shape[1]] = resized_img
        except Exception as e:
            print("e - {}, img shape: {}, ratio: {}, rshape: {}".format(
                e, img.shape, ratio, resized_img.shape))
            # cv2.imwrite("debug.png", img)
            raise Exception(e)
        
        return padded_image
    

class LossModule(nn.Module):
    def __init__(self,
                 l_coord,
                 l_noobj,
                 num_classes):
        super(LossModule, self).__init__()
        self._l_coord = l_coord
        self._l_noobj = l_noobj
        self._num_classes = num_classes
        
    def compute_iou(self, box1, box2):
        '''
        box1: N x 4, N = 2 actually
        box2: M x 4, M = 1 actually
        return Iou[N, M] where Iou(i, j) is the iou value of Ni amd Mj
        '''
        N = box1.size(0)
        M = box2.size(0)
        
        # step 1, calculate the intersection parts
        left_top = torch.max(
            box1[:, 0:2].unsqueeze(1).expand(N, M, 2), # [N 2] -> [N, 1, 2] -> [N, M, 2]
            box2[:, 0:2].unsqueeze(0).expand(N, M, 2), # [M,2] -> [1, M , 2] -> [N, M, 2]
        )
        
        right_bottom = torch.min(
            box1[:, 2:4].unsqueeze(1).expand(N, M, 2),
            box2[:, 2:4].unsqueeze(0).expand(N, M, 2)
        )
        
        inter_wh = right_bottom - left_top
        inter_wh[inter_wh < 0] = 0
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N, ) 
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M, )
        area1 = area1.unsqueeze(1).expand_as(inter_area)  # N -> N,1 -> N,M
        area2 = area2.unsqueeze(0).expand_as(inter_area)  # M -> 1,M -> N,M
        iou = inter_area / (area1 + area2 - inter_area)
        
        return iou
    
    def forward(self, pred_tensor, target_tensor):
        '''
        pred_tensor: N x S x S x (2x5+3) = N x 7 x 7 x 13
        target_tensor: N x S x S x (2x5+3) = N x 7 x 7 x 13
        '''
        N = pred_tensor.size()[0]
        positive_mask = target_tensor[:,:,:,4] > 0
        negative_mask = target_tensor[:,:,:,4] == 0
        positive_mask = positive_mask.unsqueeze(-1).expand_as(target_tensor)
        negative_mask = negative_mask.unsqueeze(-1).expand_as(target_tensor)
        
        positive_pred = pred_tensor[positive_mask].view(-1, 10 + self._num_classes)
        positive_box_pred = positive_pred[:, 0:10].contiguous().view(-1, 5)  # [n, x, y, w, h, c]
        positive_class_pred = positive_pred[:, 10:]
        
        positive_target = target_tensor[positive_mask].view(-1, 10 + self._num_classes)
        positive_box_target = positive_target[:, 0:10].contiguous().view(-1, 5)  # [n, x, y, w, h, c]
        positive_class_target = positive_target[:, 10:]
        
        # comput not contain obj loss (False positive)
        # in each grid, find the predicted boxes that should be the background
        negative_pred = pred_tensor[negative_mask].view(-1, 10 + self._num_classes)
        negative_target = target_tensor[negative_mask].view(-1, 10 + self._num_classes)
        # find the predicted object that c > 0 but should be negative
        negative_pred_mask = torch.zeros(negative_pred.size(), dtype=torch.bool)
        negative_pred_mask[:, 4] = True
        negative_pred_mask[:, 9] = True
        
        negative_pred_c = negative_pred[negative_pred_mask]
        negative_target_c = negative_target[negative_pred_mask]
        
        negative_c_loss = torch.nn.MSELoss()(negative_pred_c, negative_target_c)
        
        # compute contain obj loss (True positive and false negative)
        positive_response_mask = torch.zeros(positive_box_target.size(), dtype=torch.bool)
        positive_not_response_mask = torch.zeros(positive_box_target.size(), dtype=torch.bool)
        box_target_iou = torch.zeros(positive_box_target.size())

        # in top 2 boxes that predicted by the network in each grid, chose the
        # one with the maximum iou with the target as the responsed one
        for index in range(0, positive_box_target.size()[0], 2):
            box1 = positive_box_pred[index:index + 2] # each 2 detected box in each grid
            box1_xyxy = torch.zeros(box1.size(), dtype=torch.float32)
            # TODO: make sure the box1 [0:2] is relateve coordinate
            box1_xyxy[:, 0:2] = box1[:, 0:2] * 7 - 0.5 * box1[:, 2:4]
            box1_xyxy[:, 2:4] = box1[:, 0:2] * 7 + 0.5 * box1[:, 2:4]

            box2 = positive_box_target[index].view(-1, 5)
            box2_xyxy = torch.zeros(box2.size())
            box2_xyxy[:, 0:2] = box2[:, 0:2] * 7 - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, 0:2] * 7 + 0.5 * box2[:, 2:4]
            
            iou = self.compute_iou(box1_xyxy, box2_xyxy)
            max_iou, max_index = iou.max(0)
            positive_response_mask[index + max_index] = True   # max_index belongs 0, 1
            positive_not_response_mask[index + 1 - max_index] = True

            box_target_iou[index + max_index, 4] = max_iou

        box_pred_response = positive_box_pred[positive_response_mask].view(-1, 5)
        box_target_response = positive_box_target[positive_response_mask].view(-1, 5)
        box_response_iou = box_target_iou[positive_response_mask].view(-1, 5)
        
        # IOU loss
        iou_loss = torch.nn.MSELoss()(
            box_pred_response[:, 4], box_response_iou[:, 4])

        # positve should not contain loss
        box_pred_not_response = positive_box_pred[positive_not_response_mask].view(-1, 5)
        box_target_not_response = positive_box_target[positive_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0
        not_contain_loss = torch.nn.MSELoss()(
            box_pred_not_response[:, 4],
            box_target_not_response[:, 4]
        )

        # location loss
        loc_loss = nn.MSELoss()(box_pred_response[:, 0:2], box_target_response[:, 0:2]) + \
                   nn.MSELoss()(torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]))

        # class loss
        class_loss = nn.MSELoss()(
            positive_class_pred,
            positive_class_target
        )

        loss = self._l_coord * loc_loss + \
               2 * iou_loss + \
               not_contain_loss + self._l_noobj * negative_c_loss + \
               class_loss

        return loss

class MyTrainer:
    def __init__(self,
                 batch_size,
                 max_epoch,
                 data_dir,
                 basic_learning_rate,
                 model_save_dir,
                 num_classes,
                 warmup_epoches=5,
                 warmup_lr=0,
                 device="cpu") -> None:
        self._device = device
        self._batch_size = batch_size
        self._max_epoch = max_epoch
        self._basic_learning_rate = basic_learning_rate
        self._warmup_lr = warmup_lr
        self._warmup_epoches = warmup_epoches
        self._epoch = 0
        self._num_classes = num_classes
        self._model_name = "yolo_v1"
        
        train_data_dir = os.path.join(data_dir, "train")
        self._train_dataset = DataSet(
            train_data_dir,
            step="train",
            mean_data=np.array([131.35, 233.25, 128.46]),
            std_data=np.array([68.53, 59.17, 60.6])
        )
        self._train_data_loeader = DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True
        )
        
        valid_data_dir = os.path.join(data_dir, "valid")
        self._valid_dataset = DataSet(
            valid_data_dir,
            step="valid",
            mean_data=np.array([91.53, 93.18, 94.14]),
            std_data=np.array([56.7, 56.33, 57.49])
        )
        self._valid_data_loader = DataLoader(
            self._valid_dataset,
            batch_size=self._batch_size,
            shuffle=False
        )

        # define the module
        self._model = Yolov1(self._num_classes)
        input_size = self._train_dataset.input_dim
        print("input size: {}".format(input_size))
        summary(self._model, (3, input_size[0], input_size[1]))
        
        # before train, setup the training env
        self._optimizer = self.get_optimizer(
            self._model,
            self._basic_learning_rate,
            batch_size
        )
        self._lr_scheduler = lr_scheduler.StepLR(
            self._optimizer,
            step_size=10,
            gamma=0.2)
        
        self._loss_fn = LossModule(5, 0.5 , self._num_classes).forward
        
        # model save related
        self._model_save_dir = model_save_dir
        self._best_loss =  100000000000
        self._best_model = None
        
    def get_optimizer(self, model, basic_lr, batch_size):
        # params = []
        # params_dict = dict(model.named_parameters())
        # for name, param in params_dict.items():
        #     # print("{}: {}".format(name, param))
        #     if name.startswith("_back_bone"):
        #         params += [{"params": [param], "lr": basic_lr * 1.5}]
        #     else:
        #         params += [{"params": [param], "lr": basic_lr}]
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=basic_lr,
            weight_decay=0.0001
        )
        
        return optimizer
    
    def save_checkpoint(self, state, save_dir, model_name=""):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, model_name + "_best_ckpt.pth")
        torch.save(state, filename)
        
    def save_models(self):
        if self._best_model is None:
            print("The model is None, can not save")
        ckpt_state = {
            "start_epoch": self._epoch,
            "model": self._best_model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "best_loss": self._best_loss,
        }
        self.save_checkpoint(ckpt_state, self._model_save_dir, self._model_name)
    
    def train(self):
        self._model.to(self._device)
        self._model.train()
        
        for self._epoch in range(1, self._max_epoch + 1):
            train_total_loss = 0
            train_total_samples = 0
            self._iter = 0
            for train_data in enumerate(self._train_data_loeader):
                self._iter = train_data[0]
                inputs, targets = train_data[1]
                
                # print("inputs: {}, targets: {}".format(inputs.size(), targets.size()))
                inputs = inputs.to(torch.float32)
                outputs = self._model(inputs)
                # print("outputs: {}".format(outputs.data))
                loss = self._loss_fn(outputs, targets)
                loss.backward()
                self._optimizer.step()
                
                # upgrade learning rate
                # print("basic lr: {}".format(self._lr_scheduler.get_last_lr()))
                self._lr_scheduler.step()
                # for param_group in self._optimizer.param_groups:
                #     param_group["lr"] = new_lr
                
                train_total_loss += loss.item() * inputs.size(0)
                print("\rpoch: {}, itr: {}, loss: {:.04f}".format(
                    self._epoch,
                     self._iter, loss.item()), end=' ')
            
            print("train epoch {}, avg_loss = {:0.4f}".format(
                self._epoch,
                train_total_loss / train_total_samples
            ))

            valid_total_loss = 0
            valid_total_samples = 0
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self._valid_data_loader):
                    inputs = inputs.to(torch.float32)
                    # targets = targets.unsqueeze(1)
                    outputs = self._model(inputs)
                    # outputs = outputs.squeeze()
                    loss = self.loss_fn(self._model, outputs, targets)
                    valid_total_loss += loss.item() * inputs.size(0)
                    valid_total_samples += inputs.size(0)
            print("valid epoch {}, avg loss = {:.04f}".format(
                self._epoch, valid_total_loss / valid_total_samples
            ))
            
            if valid_total_loss / valid_total_samples < self._best_loss:
                self._best_loss = valid_total_loss / valid_total_samples
                self._best_model = self._model
                self.save_models()


if __name__ == "__main__":
    trainer = MyTrainer(
        batch_size=2,
        max_epoch=50,
        data_dir="/home/lutao/datasets/face_mask_data/face_mask_v0",
        basic_learning_rate=0.001,
        model_save_dir="./save_models/yolov1",
        num_classes=3,
    )
    trainer.train()
