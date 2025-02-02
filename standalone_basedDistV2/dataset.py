from __future__ import print_function, division
import os
from scipy.optimize import curve_fit
import warnings
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import argparse
from skimage import io
import pandas as pd
import torch
import math
import numpy as np
import itertools


os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
warnings.filterwarnings("ignore")


class RadioUNet_s(Dataset):

    def __init__(self, client_id=0, clients_num=10, phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="../../../Datasets/RadioMapSeer4FL/",    # path to dataset
                 numTx=80*49,  # !
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 IRT2maxW=1,
                 missing=1,
                 fix_samples=655*10,
                 num_samples_low=655,
                 num_samples_high=655*10,
                 transform=transforms.ToTensor()):
        """
        Outputs:
            1.inputs: image_buildings, image_Tx, image_samples, genImg
            2.image_gain
        """
        super(RadioUNet_s, self).__init__()
        self.client_id = client_id
        self.clients_num = clients_num
        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh
        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        self.arr = np.arange(64)  # ! 64
        self.one = np.ones(64)
        self.img = np.outer(self.arr, self.one)
        self.IRT2maxW = IRT2maxW
        self.cityMap = cityMap
        self.missing = missing
        self.fix_samples = fix_samples
        self.num_samples_low = num_samples_low
        self.num_samples_high = num_samples_high
        self.transform = transform
        self.height = 64  # ! 64
        self.width = 64

        num_train = int(self.numTx * 0.95)
        if phase == "train":
            self.ind1 = 0
            self.ind2 = num_train-1
        elif phase == "test":
            self.ind1 = num_train
            self.ind2 = self.numTx-1
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2-1

        # Directories for buildings, antennas, and gains
        self.dir_Tx = self.dir_dataset + "png/antennas/"
        if self.simulation == "DPM":
            if self.carsSimul == "no":
                self.dir_gain = self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset+"gain/carsDPM/"
        elif self.simulation == "IRT2":
            if self.carsSimul == "no":
                self.dir_gain = self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset+"gain/carsIRT2/"
        elif self.simulation == "rand":
            if self.carsSimul == "no":
                self.dir_gainDPM = self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2 = self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM = self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2 = self.dir_dataset+"gain/carsIRT2/"

        if self.cityMap == "complete":
            self.dir_buildings = self.dir_dataset+"png/buildings_complete/"
            self.dir_genimg = self.dir_dataset+"png/genImg_655based/"
        else:
            # a random index will be concatenated in the code
            self.dir_buildings = self.dir_dataset+"png/buildings_missing"
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        # later check if reading the JSON file and creating antenna images on the fly is faster
        if self.carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

    def __len__(self):
        return (self.ind2-self.ind1+1)

    def __getitem__(self, idx):
        idxr = np.floor(idx/self.numTx).astype(int)
        idxc = idx-idxr*self.numTx
        map_index = [0, 449, 77, 78, 88, 94, 97, 109, 153, 179, 291, 341, 458]
        dataset_map_ind = map_index[self.client_id]
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        # name2 = str(dataset_map_ind) + "_" + str(idx+self.ind1) + ".png"
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        name3 = str(dataset_map_ind) + "_" + str(idxc) + ".npy"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name2)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(
                self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name2)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))/256

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(
                io.imread(img_name_gain)), axis=2)/256
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            # IRT2 weight of random average
            w = np.random.uniform(0, self.IRT2maxW)
            image_gain = w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2)/256  \
                + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2)/256
        # load genimg based on 655 samples
        img_genImg = os.path.join(self.dir_genimg, name3)
        genImg = np.asarray(np.load(img_genImg))
        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain = image_gain/(1-self.thresh)

        # we use this normalization so all RadioUNet methods can have the same learning rate.
        image_gain = image_gain*256

        # input measurements
        image_samples = np.zeros((64, 64))  # !256 -> 64

        # num_samples = np.floor(self.fix_samples).astype(int)
        # x_samples = np.random.randint(0, 63, size=num_samples)
        # y_samples = np.random.randint(0, 63, size=num_samples)
        if self.fix_samples == 0:
            num_samples = np.random.randint(
                self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_samples = np.floor(self.fix_samples).astype(int)
        x_samples = np.random.randint(0, 63, size=num_samples)  # !
        y_samples = np.random.randint(0, 63, size=num_samples)  # !
        if self.fix_samples == 1:
            side = np.random.randint(0, 2)
            if side == 1:
                x_samples = np.append(np.random.randint(
                    0, 32, size=1600), np.random.randint(32, 63, size=160))
                y_samples = np.random.randint(0, 63, size=1600+160)
            else:
                x_samples = np.append(np.random.randint(
                    0, 32, size=160), np.random.randint(32, 63, size=1600))
                y_samples = np.random.randint(0, 63, size=1600+160)

        image_samples[x_samples,
                      y_samples] = image_gain[x_samples, y_samples, 0]

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx,
                              image_samples, genImg], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            # note that ToTensor moves the channel from the last asix to the first!

        return [inputs, image_gain]

    def get_map_size(self):
        return 256

    def get_numTx(self):
        return self.numTx


def DataModule(clients_num, batchSize, simulation, carsSimulation, secarsInput, cityMap, num_workers, fix, low, high, mode='train'):

    dataloaders = []
    if mode == 'train':
        for client_id in range(clients_num):
            original_train = RadioUNet_s(client_id=client_id, clients_num=clients_num, phase="train", simulation=simulation,
                                         carsSimul=carsSimulation, carsInput=secarsInput, cityMap=cityMap, fix_samples=fix, num_samples_low=low, num_samples_high=high)

            train_loader = DataLoader(original_train, batchSize,
                                      shuffle=True, num_workers=num_workers, drop_last=False)
            dataloaders.append(train_loader)
    elif mode == 'test':
        for client_id in range(clients_num):
            original_test = RadioUNet_s(client_id=client_id, clients_num=clients_num, phase="test", simulation=simulation,
                                        carsSimul=carsSimulation, carsInput=secarsInput, cityMap=cityMap, fix_samples=fix, num_samples_low=low, num_samples_high=high)
            test_loader = DataLoader(original_test, len(original_test),
                                     shuffle=False, num_workers=num_workers, drop_last=False)
            dataloaders.append(test_loader)

    return dataloaders


class FederatedRadioUNet(Dataset):
    def __init__(self, original_dataset, submap_size, client_id, num_clients=9):  # num_clients=16
        self.original_dataset = original_dataset
        self.client_id = client_id
        self.map_size = self.original_dataset.get_map_size()
        self.kernel_size = submap_size  # 64
        self.stride = int((self.map_size-submap_size) //
                          (math.sqrt(num_clients)-1))  # 64

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        maps = self.original_dataset[idx]
        applied_maps = []
        for map in maps:

            maps_train = []
            _, height, width = map.shape

            for i, j in itertools.product(
                    range(0, height - self.kernel_size + 1, self.stride),
                    range(0, width - self.kernel_size + 1, self.stride)):
                window = map[:, i:i + self.kernel_size, j:j+self.kernel_size]
                maps_train.append(window)
            applied_maps.append(maps_train)
        return [applied_maps[0][self.client_id], applied_maps[1][self.client_id]]


def merge_federated_maps(federated_maps, original_size=256, num_clients=9):
    """
    Merge maps from federated clients into a single map.

    :param federated_maps: List of maps from each client, shape (num_clients, channels, submap_size, submap_size)
    :param original_size: Size of the original map before splitting
    :param num_clients: Number of clients (should be a perfect square)
    :return: Merged map of shape (channels, original_size, original_size)
    """
    num_clients_sqrt = int(math.sqrt(num_clients))
    submap_size = federated_maps[0].shape[-1]  # Assuming square submaps

    stride = int((original_size - submap_size) // (num_clients_sqrt - 1))

    # Initialize the merged map
    channels = federated_maps[0].shape[1]
    batch = federated_maps[0].shape[0]
    merged_map = np.zeros((batch, channels, original_size, original_size))

    for i in range(num_clients_sqrt):
        for j in range(num_clients_sqrt):
            client_idx = i * num_clients_sqrt + j
            start_h = i * stride
            start_w = j * stride
            merged_map[:, :, start_h:start_h+submap_size,
                       start_w:start_w+submap_size] = federated_maps[client_idx]

    return merged_map


def merge_eachClient_maps(federated_maps, original_size=256, num_splited=49):
    """
    Merge maps into a single map for each client.

    :param federated_maps: List of maps from each client, shape (num_clients, channels, submap_size, submap_size)
    :param original_size: Size of the original map before splitting
    :param num_splited: Number of splited submaps (should be a perfect square)
    :return: Merged map of shape (channels, original_size, original_size)
    """
    num_clients_sqrt = int(math.sqrt(num_splited))  # 7
    # [num_clients, num_submaps, b, submap_size, c] = np.array(
    #     federated_maps).shape
    num_clients = len(federated_maps)
    submap_size = 64
    num_submaps = 196

    stride = int((original_size - submap_size) // (num_clients_sqrt - 1))  # 32

    num_maps = int(num_submaps/num_splited)  # 4
    merged_maps_clients = []
    for n in range(num_clients):
        for o in range(num_maps):
            m = 0
            merged_maps = []
            merged_map = np.zeros((original_size, original_size))
            for i in range(num_clients_sqrt):
                for j in range(num_clients_sqrt):
                    start_h = i * stride
                    start_w = j * stride
                    merged_map[start_h:start_h+submap_size,
                               start_w:start_w+submap_size] = federated_maps[n][m][0]
                    m += 80  # !
            merged_maps.append(merged_map)
        merged_maps_clients.append(merged_maps)
    return merged_maps_clients  # (9,4,256,256)


def merge_with_overlap(federated_maps, original_size=256, num_clients=9):
    """
    Merge maps from federated clients into a single map, averaging overlapping areas.

    :param federated_maps: List of maps from each client, shape (num_clients, channels, submap_size, submap_size)
    :param original_size: Size of the original map before splitting
    :param num_clients: Number of clients (should be a perfect square)
    :return: Merged map of shape (channels, original_size, original_size)
    """
    num_clients_sqrt = int(math.sqrt(num_clients))
    submap_size = federated_maps[0].shape[-1]  # Assuming square submaps
    stride = int((original_size - submap_size) // (num_clients_sqrt - 1))

    # Initialize the merged map and a count map to track overlaps
    channels = federated_maps.shape[0]
    merged_map = np.zeros((channels, original_size, original_size))
    count_map = np.zeros((original_size, original_size))

    for i in range(num_clients_sqrt):
        for j in range(num_clients_sqrt):
            client_idx = i * num_clients_sqrt + j
            start_h = i * stride
            start_w = j * stride

            # Add the submap to the merged map
            merged_map[:, start_h:start_h+submap_size,
                       start_w:start_w+submap_size] = federated_maps[client_idx]

            # Increment the count for this region
            count_map[start_h:start_h+submap_size,
                      start_w:start_w+submap_size] = 1

    # Average the overlapping areas
    for c in range(channels):
        overlap_mask = count_map >= 1
        merged_map[c, overlap_mask] /= count_map[overlap_mask]
    return merged_map
