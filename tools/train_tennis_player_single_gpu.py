import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", 'val')
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def main(args):

    # Register datasets
    register_coco_instances("usopen_nadal_train",
                        {}, 
                        "./data/usopen_nadal_v1/annotations_train.json",
                        "./data/usopen_nadal_v1/images")

    register_coco_instances("usopen_nadal_val",
                        {}, 
                        "./data/usopen_nadal_v1/annotations_val.json",
                        "./data/usopen_nadal_v1/images")


    register_coco_instances("usopen_nadal_test",
                        {}, 
                        "./data/usopen_nadal_test2/annotations.json",
                        "./data/usopen_nadal_test2/images")


    cfg = get_cfg()
    cfg.merge_from_file(
        "./configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        # "./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )

    cfg.INPUT.MASK_FORMAT = "bitmask"  
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASETS.TRAIN = ("usopen_nadal_train",)
    cfg.DATASETS.TEST = ("usopen_nadal_val",)  

    cfg.MODEL.WEIGHTS = "./models/mask_rcnn_R_101_FPN_3x.pkl"  # initialize from model zoo
    # cfg.MODEL.WEIGHTS = "./models/mask_rcnn_X_101_32x8d_FPN_3x.pkl"  # initialize from model zoo
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # cfg.SOLVER.REFERENCE_WORLD_SIZE = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    # cfg.SOLVER.MAX_ITER = 1000 
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000

    cfg.TEST.EVAL_PERIOD = 2000

    # cfg.OUTPUT_DIR = './output/usopen_nadal_v1'
    cfg.OUTPUT_DIR = './output/usopen_nadal_test2_R101'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Inference
    if args.eval_only:
        cfg.DATASETS.TEST = ("usopen_nadal_test",)  
        cfg.freeze()
        default_setup(cfg, args)

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        evaluator = Trainer.build_evaluator(
            cfg,
            dataset_name=cfg.DATASETS.TEST[0],
            output_folder=os.path.join(cfg.OUTPUT_DIR, "inference", "test")
            )
        res = Trainer.test(cfg, model, evaluator)

        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return

    # Train
    cfg.freeze()
    default_setup(cfg, args)
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)