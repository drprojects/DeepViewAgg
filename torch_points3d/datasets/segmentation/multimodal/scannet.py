from ..scannet import *
from torch_points3d.datasets.base_dataset_multimodal import BaseDatasetMM


log = logging.getLogger(__name__)

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
