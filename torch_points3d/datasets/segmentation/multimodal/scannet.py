from ..scannet import *
from .utils import read_image_pose_pairs
from torch_points3d.core.multimodal.image import SameSettingImageData
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM


log = logging.getLogger(__name__)


########################################################################################
#                                                                                      #
#                            ScanNet image processing utils                            #
#                                                                                      #
########################################################################################

def load_pose(filename):
    """Read ScanNet pose file.
    Credit: https://github.com/angeladai/3DMV/blob/f889b531f8813d409253fe065fb9b18c5ca2b495/3dmv/data_util.py
    """
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


########################################################################################
#                                                                                      #
#                                 ScanNet 2D-3D dataset                                #
#                                                                                      #
########################################################################################

class ScannetMM(Scannet):
    DEPTH_IMG_SIZE = (640, 480)

    def __init__(self, *args, img_ref_size=(320, 240), **kwargs):
        self.img_ref_size = img_ref_size
        super(ScannetMM, self).__init__(*args, **kwargs)

    def process(self):
        if self.is_test:
            return

        # --------------------------------------------------------------
        # Preprocess 3D data
        # Download, pre_transform and pre_filter raw 3D data
        # Output will be saved to preprocessed_<SPLIT>.pt
        # --------------------------------------------------------------
        super().process()

        # --------------------------------------------------------------
        # Recover image data
        # --------------------------------------------------------------
        # TODO: Multimodal pre_transform here
        #  - decide file structure for image_data
        #  - build the image_data list, without mappings
        #  - compute the mappings

        for i, (scan_names, split) in enumerate(zip(self.scan_names, self.SPLITS)):
            if not osp.exists(self.image_data_paths[i]):
                mapping_idx_to_scan_names = getattr(self, "MAPPING_IDX_TO_SCAN_{}_NAMES".format(split.upper()))
                scannet_dir = osp.join(self.raw_dir, "scans" if split in ["train", "val"] else "scans_test")

                for id, scan_name in enumerate(scan_names):
                    scan_sens_dir = osp.join(scannet_dir, scan_name, 'sens')
                    meta_file = osp.join(scannet_dir, scan_name, scan_name + ".txt")

                    # Recover the image-pose pairs
                    image_info_list = [
                        {'path': i_file, 'extrinsic': load_pose(p_file)}
                        for i_file, p_file in read_image_pose_pairs(
                            osp.join(scan_sens_dir, 'color'),
                            osp.join(scan_sens_dir, 'pose'),
                            image_suffix='.jpg', pose_suffix='.txt')]

                    # Aggregate all RGB image paths
                    path = np.array([info['path'] for info in image_info_list])

                    # Aggregate all extrinsic 4x4 matrices
                    # Train and val scans have undergone axis alignment
                    # transformations. Need to recover and apply those
                    # to camera poses too. Test scans have no axis
                    # alignment
                    axis_align_matrix = read_axis_align_matrix(meta_file)
                    if axis_align_matrix is not None:
                        extrinsic = torch.cat([
                            axis_align_matrix.mm(info['extrinsic']).unsqueeze(0)
                            for info in image_info_list], dim=0)
                    else:
                        extrinsic = torch.cat([
                            info['extrinsic'].unsqueeze(0)
                            for info in image_info_list], dim=0)

                    # For easier image handling, extract the images
                    # position from the extrinsic matrices
                    xyz = extrinsic[:, :3, 3]

                    # Read intrinsic parameters for the depth camera
                    # because this is the one related to the extrinsic.
                    # Strangely, using the color camera intrinsic along
                    # with the pose does not produce the expected
                    # projection
                    intrinsic = load_pose(osp.join(scan_sens_dir, 'intrinsic/intrinsic_depth.txt'))
                    fx = intrinsic[0][0].repeat(len(image_info_list))
                    fy = intrinsic[1][1].repeat(len(image_info_list))
                    mx = intrinsic[0][2].repeat(len(image_info_list))
                    my = intrinsic[1][2].repeat(len(image_info_list))

                    # Save scan images as SameSettingImageData
                    # NB: the image is first initialized to
                    # DEPTH_IMG_SIZE because the intrinsic parameters
                    # are defined accordingly. Setting ref_size
                    # afterwards calls the @adjust_intrinsic method to
                    # rectify the intrinsics consequently
                    image_data = SameSettingImageData(
                        ref_size=self.DEPTH_IMG_SIZE, proj_upscale=1, path=path,
                        pos=xyz, fx=fx, fy=fy, mx=mx, my=my,
                        extrinsic=extrinsic)
                    image_data.ref_size = self.img_ref_size

                    # TODO: WHAT NEXT ?






    def _init_load(self, split):
        if split == "train":
            path = self.processed_paths[3]
        elif split == "val":
            path = self.processed_paths[4]
        elif split == "test":
            path = self.processed_paths[5]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, or test"))
        self.data, self.images, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return [f"preprocessed_{s}.pt" for s in Scannet.SPLITS] + [f"{s}.pt" for s in Scannet.SPLITS]

    @property
    def image_data_paths(self):
        return [osp.join(self.processed_dir, f"image_data_{s}.pt") for s in Scannet.SPLITS]

    @property
    def pre_transformed_image_path(self):
        return osp.join(self.processed_dir, "pre_transform_image.pt")



class ScannetDatasetMM(BaseDatasetMM):
    """ Wrapper around Scannet that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - dataroot
            - version
            - max_num_point (optional)
            - use_instance_labels (optional)
            - use_instance_bboxes (optional)
            - donotcare_class_ids (optional)
            - pre_transforms (optional)
            - train_transforms (optional)
            - val_transforms (optional)
    """

    SPLITS = SPLITS

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        use_instance_labels: bool = dataset_opt.use_instance_labels
        use_instance_bboxes: bool = dataset_opt.use_instance_bboxes
        donotcare_class_ids: [] = list(dataset_opt.get('donotcare_class_ids', []))
        max_num_point: int = dataset_opt.get('max_num_point', None)
        process_workers: int = dataset_opt.process_workers if hasattr(dataset_opt, 'process_workers') else 0
        is_test: bool = dataset_opt.get('is_test', False)
        types = [".sens", ".txt", "_vh_clean_2.ply", "_vh_clean_2.0.010000.segs.json", ".aggregation.json"]

        self.train_dataset = Scannet(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
            types=types,
            frame_depth=True,
            frame_rgb=True,
            frame_pose=True,
            frame_intrinsics=True,
            frame_skip=100
        )

        self.val_dataset = Scannet(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
            types=types,
            frame_depth=True,
            frame_rgb=True,
            frame_pose=True,
            frame_intrinsics=True,
            frame_skip=100
        )

        self.test_dataset = Scannet(
            self._data_path,
            split="test",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            version=dataset_opt.version,
            use_instance_labels=use_instance_labels,
            use_instance_bboxes=use_instance_bboxes,
            donotcare_class_ids=donotcare_class_ids,
            max_num_point=max_num_point,
            process_workers=process_workers,
            is_test=is_test,
            types=types,
            frame_depth=True,
            frame_rgb=True,
            frame_pose=True,
            frame_intrinsics=True,
            frame_skip=100
        )

    @property
    def path_to_submission(self):
        return self.train_dataset.path_to_submission

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            dataset {[type]}
            wandb_log - Log using weight and biases
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.scannet_segmentation_tracker import ScannetSegmentationTracker

        return ScannetSegmentationTracker(
            self, wandb_log=wandb_log, use_tensorboard=tensorboard_log, ignore_label=IGNORE_LABEL
        )


########################################################################################
#                                                                                      #
#                          Script to load a few ScanNet scans                          #
#                                                                                      #
########################################################################################


if __name__ == '__main__':

    from omegaconf import OmegaConf

    # Recover dataset options
    DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    dataset_options = OmegaConf.load(os.path.join(DIR,'conf/data/multimodal/scannet.yaml'))

    # Choose download root directory
    dataset_options.data.dataroot = os.path.join(DIR,"data")
    reply = input(f"Save dataset to {dataset_options.data.dataroot} ? [y/n] ")
    if reply.lower() == 'n':
        dataset_options.data.dataroot = ""
        while not osp.exists(dataset_options.data.dataroot):
            dataset_options.data.dataroot = input(f"Please provide an existing directory to which the dataset should be dowloaded : ")

    # Download the hard-coded release scans 
    dataset = ScannetDatasetMM(dataset_options.data)
    print(dataset)
