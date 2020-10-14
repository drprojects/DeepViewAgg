import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
from torch_geometric.datasets import S3DIS as S3DIS1x1
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
import json
from PIL import Image
from functools import partial

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.multimodal.base_dataset import BaseDatasetMM

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

S3DIS_NUM_CLASSES = 13

INV_OBJECT_LABEL = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'ceiling' .-> .yellow
        [95, 156, 196],  # 'floor' .-> . blue
        [179, 116, 81],  # 'wall'  ->  brown
        [241, 149, 131],  # 'beam'  ->  salmon
        [81, 163, 148],  # 'column'  ->  bluegreen
        [77, 174, 84],  # 'window'  ->  bright green
        [108, 135, 75],  # 'door'   ->  dark green
        [41, 49, 101],  # 'chair'  ->  darkblue
        [79, 79, 76],  # 'table'  ->  dark grey
        [223, 52, 52],  # 'bookcase'  ->  red
        [89, 47, 95],  # 'sofa'  ->  purple
        [81, 109, 114],  # 'board'   ->  grey
        [233, 233, 229],  # 'clutter'  ->  light grey
        [0, 0, 0],  # unlabelled .->. black
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

ROOM_TYPES = {
    "conferenceRoom": 0,
    "copyRoom": 1,
    "hallway": 2,
    "office": 3,
    "pantry": 4,
    "WC": 5,
    "auditorium": 6,
    "storage": 7,
    "lounge": 8,
    "lobby": 9,
    "openspace": 10
}

VALIDATION_ROOMS = [
    "hallway_1",
    "hallway_6",
    "hallway_11",
    "office_1",
    "office_6",
    "office_11",
    "office_16",
    "office_21",
    "office_26",
    "office_31",
    "office_36",
    "WC_2",
    "storage_1",
    "storage_5",
    "conferenceRoom_2",
    "auditorium_1",
]

################################### UTILS #######################################


def object_name_to_label(object_class):
    """convert from object name in S3DIS to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label


def read_s3dis_format(train_file, room_name, label_out=True, verbose=False, debug=False):
    """extract data from a room folder"""

    room_type = room_name.split("_")[0]
    room_label = ROOM_TYPES[room_type]
    raw_path = osp.join(train_file, "{}.txt".format(room_name))
    if debug:
        reader = pd.read_csv(raw_path, delimiter="\n")
        RECOMMENDED = 6
        for idx, row in enumerate(reader.values):
            row = row[0].split(" ")
            if len(row) != RECOMMENDED:
                log.info("1: {} row {}: {}".format(raw_path, idx, row))

            try:
                for r in row:
                    r = float(r)
            except:
                log.info("2: {} row {}: {}".format(raw_path, idx, row))

        return True
    else:
        room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float32")
        try:
            rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
        except ValueError:
            rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
            log.warning("WARN - corrupted rgb data for file %s" % raw_path)
        if not label_out:
            return xyz, rgb
        n_ver = len(room_ver)
        del room_ver
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
        room_labels = np.zeros((n_ver,), dtype="int64")
        room_label = np.asarray([room_label])
        instance_labels = np.zeros((n_ver,), dtype="int64")
        objects = glob.glob(osp.join(train_file, "Annotations/*.txt"))
        i_object = 1
        for single_object in objects:
            object_name = os.path.splitext(os.path.basename(single_object))[0]
            if verbose:
                log.debug("adding object " +
                          str(i_object) + " : " + object_name)
            object_class = object_name.split("_")[0]
            object_label = object_name_to_label(object_class)
            obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
            _, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
            room_labels[obj_ind] = object_label
            instance_labels[obj_ind] = i_object
            i_object = i_object + 1

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
            torch.from_numpy(room_labels),
            torch.from_numpy(instance_labels),
            torch.from_numpy(room_label),
        )


def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "S3DIS")
    PlyData([el], byte_order=">").write(file)


def add_weights(dataset, train, class_weight_method):
    L = len(INV_OBJECT_LABEL.keys())
    if train:
        weights = torch.ones(L)
        if class_weight_method is not None:

            idx_classes, counts = torch.unique(dataset.data.y, return_counts=True)

            dataset.idx_classes = torch.arange(L).long()
            weights[idx_classes] = counts.float()
            weights = weights.float()
            weights = weights.mean() / weights
            if class_weight_method == "sqrt":
                weights = torch.sqrt(weights)
            elif str(class_weight_method).startswith("log"):
                weights = 1 / torch.log(1.1 + weights / weights.sum())

            weights /= torch.sum(weights)
        log.info(
            "CLASS WEIGHT : {}".format(
                {name: np.round(weights[index].item(), 4)
                 for index, name in INV_OBJECT_LABEL.items()}
            )
        )
        setattr(dataset, "weight_classes", weights)
    else:
        setattr(dataset, "weight_classes", torch.ones((len(INV_OBJECT_LABEL.keys()))))

    return dataset


def read_s3dis_pose(json_file):
    
    # Area 5b poses need a special treatment
    # Need to see the file comes from Area i in the provided filepath
    area_5b = 'area_5b' in json_file.lower()
    
    # Loading the Stanford pose json file
    with open(json_file) as f:
        pose_data = json.load(f)
    
    # XYZ camera position
    xyz = np.array(pose_data['camera_location'])
    
    # Omega, Phi, Kappa camera pose
    # We define a different pose coordinate system 
    omega, phi, kappa = [np.double(i) for i in pose_data['final_camera_rotation']]
#     opk = np.array([omega - (np.pi / 2), -phi, -kappa])
    opk = np.array([omega - (np.pi / 2), -phi, -kappa - (np.pi / 2)])

    # Area 5b poses require some rotation and offset corrections
    if area_5b:
        M = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])
        xyz = M.dot(xyz) + np.array([-4.10, 6.25, 0.0])
        angles = opk + np.array([0, 0, np.pi/2])
    
    return xyz, opk


def non_static_pixels_mask(img_list, img_size=(1024,512), sample=5):
    """Find the mask of identical pixels accross a list of images."""
    
    # Iteratively update the mask w.r.t. a reference image
    mask = np.ones(img_size, dtype='bool')
    img_1 = np.array(Image.open(img_list[0]).convert('RGB').resize(img_size, Image.LANCZOS))
        
    for img_path in np.random.choice(img_list[1:], size=min(sample, len(img_list)-1), replace=False):
        img_2 = np.array(Image.open(img_path).convert('RGB').resize(img_size, Image.LANCZOS))
        
        mask_equal = np.all(img_1 == img_2, axis=2).transpose()
        mask[np.logical_and(mask, mask_equal)] = 0
    
    return mask 


def image_room(image_path):
    return '_'.join(osp.basename(image_path).split('_')[2:4])



################################### Used for fused s3dis radius sphere ###################################


class S3DISOriginalFused(InMemoryDataset):
    """ Original S3DIS dataset. Each area is loaded individually and can be processed using a pre_collate transform. 
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single room by default.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    form_url = "https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1"
    download_url = "https://drive.google.com/uc?id=0BweDykwS9vIobkVPN0wzRzFwTDg&export=download"
    zip_name = "Stanford3dDataset_v1.2_Version.zip"
    path_file = osp.join(DIR, "s3dis.patch")
    file_name = "Stanford3dDataset_v1.2"
    folders = ["Area_{}".format(i) for i in range(1, 7)]
    num_classes = S3DIS_NUM_CLASSES

    def __init__(
        self,
        root,
        test_area=6,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        projection_image=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        assert test_area in list(range(1,7))
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.projection_image = projection_image
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        super(S3DISOriginalFused, self).__init__(
            root, transform, pre_transform, pre_filter)
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
        else:
            raise ValueError(f"Split {split} found, but expected either ",
                "train, val, trainval or test")
        self._load_data(path)

        if split == "test":
            self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return self.folders

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(6)]

    @property
    def processed_file_names(self):
        test_area = self.test_area
        return (
            ["{}_{}.pt".format(s, test_area)
             for s in ["train", "val", "test", "trainval"]]
            + self.raw_areas_paths
            + [self.pre_processed_path]
        )

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    def download(self):
        raw_folders = os.listdir(self.raw_dir)
        if len(raw_folders) == 0:
            if not osp.exists(osp.join(self.root, self.zip_name)):
        ########################################################################
        # Here download and unzip images
        ########################################################################
                log.info("WARNING: You are downloading S3DIS dataset")
                log.info("Please, register yourself by filling up the form at {}".format(
                    self.form_url))
                log.info("***")
                log.info("Press any key to continue, or CTRL-C to exit. By ",
                    "continuing, you confirm filling up the form.")
                input("")
                gdown.download(self.download_url, osp.join(self.root, self.zip_name), quiet=False)
            extract_zip(osp.join(self.root, self.zip_name), self.root)
            shutil.rmtree(self.raw_dir)
            os.rename(osp.join(self.root, self.file_name), self.raw_dir)   
            shutil.copy(self.path_file, self.raw_dir)
            cmd = "patch -ruN -p0 -d  {} < {}".format(self.raw_dir, osp.join(self.raw_dir, "s3dis.patch"))
            os.system(cmd) 
        else:
            intersection = len(set(self.folders).intersection(set(raw_folders)))
            if intersection != 6:
                shutil.rmtree(self.raw_dir)
                os.makedirs(self.raw_dir)
                self.download()  

    def process(self):
        if not osp.exists(self.pre_processed_path):
            train_areas = [f for f in self.folders if str(self.test_area) not in f]
            test_areas = [f for f in self.folders if str(self.test_area) in f]

            train_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in train_areas
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if osp.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            #############################
            # Link image & pose files to the split rooms
            # train_images = 
            #############################

            test_files = [
                (f, room_name, osp.join(self.raw_dir, f, room_name))
                for f in test_areas
                for room_name in os.listdir(osp.join(self.raw_dir, f))
                if osp.isdir(osp.join(self.raw_dir, f, room_name))
            ]

            #############################
            # Link image & pose files to the split rooms
            # test_images = 
            #############################

            # Gather all data from each area in a List(List(Data))
            data_list = [[] for _ in range(6)]
            if self.debug:
                areas = np.zeros(7)
            for (area, room_name, file_path) in tq(train_files + test_files):
                if self.debug:
                    area_idx = int(area.split('_')[-1])
                    if areas[area_idx] == 5:
                        continue
                    else:
                        print(area_idx)
                        areas[area_idx] += 1
                
                area_num = int(area[-1]) - 1
                if self.debug:
                    read_s3dis_format(file_path, room_name, label_out=True, verbose=self.verbose,
                        debug=self.debug)
                    continue
                else:
                    xyz, rgb, room_labels, instance_labels, room_label = read_s3dis_format(
                        file_path, room_name, label_out=True, verbose=self.verbose,
                        debug=self.debug)

                    rgb_norm = rgb.float() / 255.0
                    data = Data(pos=xyz, y=room_labels, room_label=room_label, rgb=rgb_norm)
                    
                    if room_name in VALIDATION_ROOMS:
                        data.validation_set = True
                    else:
                        data.validation_set = False

                    if self.keep_instance:
                        data.instance_labels = instance_labels

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    data_list[area_num].append(data)

            raw_areas = cT.PointCloudFusion()(data_list)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    for data in area_datas:
                        data = self.pre_transform(data)

            # Save the data into one big 'preprocessed.pt' file 
            torch.save(data_list, self.pre_processed_path)

        else:
            # Recover the per-area Data list from the 'preprocessed.pt' file
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        # Build the train, val and trainval data lists
        train_data_list = {}
        val_data_list = {}
        trainval_data_list = {}
        for i in range(6):
            if i != self.test_area - 1:
                train_data_list[i] = []
                val_data_list[i] = []
                for data in data_list[i]:
                    validation_set = data.validation_set
                    del data.validation_set
                    if validation_set:
                        val_data_list[i].append(data)
                    else:
                        train_data_list[i].append(data)
                trainval_data_list[i] = val_data_list[i] + train_data_list[i]

        train_data_list = list(train_data_list.values())
        val_data_list = list(val_data_list.values())
        trainval_data_list = list(trainval_data_list.values())
        test_data_list = data_list[self.test_area - 1]

        # Run the pre_collate_transform to finalize the data preparation
        # Among other things, the 'origin_id' and 'preprocessed_id' are
        # generated here  
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        ######################################################################## 
        # MAPPING COMPUTATION HERE

        # Build the train, val, trainval and test image-pose lists

        # Build the mappings for each split, for each area

        # Save data splits mappings as soon as they are computer to spare memory
        # self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

        # self.projection_image(xyz_to_img, indices, img_opk, img_mask=None)
        ########################################################################


    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        """
        Save the preprocessed data lists. Results are intended to be loaded with
        self._load_data().

        #### SAVE MULTIMODAL DATA IN MEMORY HERE

        Overwrites the S3DISOriginalFused._save_data() which made use of a
        collate function.
        """
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class S3DISSphere(S3DISOriginalFused):
    """ Small variation of S3DISOriginalFused that allows random sampling of spheres 
    within an Area during training and validation. Spheres have a radius of 2m. If sample_per_epoch is not specified, spheres
    are taken on a 2m grid.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 6 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=2, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=radius / 10.0)
        super().__init__(root, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def get(self, idx):
        """
        Get a 3D points Data sample. Does not return multimodal
        attributes.

        Overwrites the torch_geometric.InMemoryDataset.get(), which is called
        from inside the torch_geometric.InMemoryDataset.__getitem__() used for 
        indexing datasets.
        """

        ########################################################################
        """ rewrite this to call multimodal items
        
        get_random 
            called if S3DISSphere is NOT test set
            S3DISSphere has predefined centers accross all areas in the split
            randomly picks a center
            samples the corresponding S3DISSphere._datas[i_area] accordingly

            Need to:
                inform of the chosen area to get the appropriate ._datas_multimodal


        """
        ########################################################################

        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def __getitem__(self, idx):
        """
        Get a 3D points Data sample and its multimodal attributes.

        Only supports indexing with int.

        #### GET MULTIMODAL DATA FROM preprocessed_id HERE

        Overwrites the torch_geometric.InMemoryDataset.__getitem__() used for 
        indexing datasets.
        """
        assert isinstance(idx, int), (f"Indexing with {type(idx)} is not ",
            f"supported, only {type(int)} are accepted.")

        # Get the 3D point sample and apply transforms
        data = self.get(self.indices()[idx])
        data = data if self.transform is None else self.transform(data)

        # Recover the multimodal attributes from the 'preprocessed_id' attribute
        assert hasattr(data, 'preprocessed_id'), ("Data has no 'preprocessed_id' attribute ",
            "required for recovering the transformed points multimodal attributes.")

        #### GET MULTIMODAL DATA FROM preprocessed_id HERE

        return data

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        super().process()

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(
            self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _load_data(self, path):
        """
        Initializes the self._datas which hold the all preprocessed data in
        memory. Also initializes the sphere sampling centers and per-area
        KDTrees.

        #### LOAD MULTIMODAL DATA IN MEMORY HERE 
        
        Overwrites the S3DISOriginalFused._load_data()
        """
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(data, cT.SphereSampling.KDTREE_KEY)  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]),
                return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)
   


class S3DISFusedDataset(BaseDatasetMM):
    """ Wrapper around S3DISSphere that creates train and test datasets.

    http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get('sampling_format', 'sphere')
        assert sampling_format == 'sphere', f"Only sampling format 'sphere' is supported."




        # projection_image = partial(compute_index_map, **self.dataset_opt.projection_image)




        self.train_dataset = S3DISSphere(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
            projection_image=projection_image,
        )

        self.val_dataset = S3DISSphere(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
            projection_image=projection_image,
        )
        self.test_dataset = S3DISSphere(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
            projection_image=projection_image,
        )

        if dataset_opt.class_weight_method:
            self.train_dataset = add_weights(
                self.train_dataset, True, dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save s3dis predictions to disk using s3dis color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.s3dis_tracker import S3DISTracker

        return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
