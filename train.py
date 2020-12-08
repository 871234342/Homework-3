import os
import matplotlib.pyplot as plt

import cv2
import json
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from itertools import groupby
from pycocotools import mask as maskutil

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def show_example(dataset_dicts, metadata, num=3):
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


setup_logger()
train_path = "train_images/"
json_file = "pascal_train.json"
register_coco_instances("VOC_dataset", {}, json_file, train_path)
dataset_dicts = load_coco_json(json_file, train_path, "VOC_dataset")

VOC_metadata = MetadataCatalog.get("VOC_dataset")
show_example(dataset_dicts, VOC_metadata, 1)

# config
model_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.DATASETS.TRAIN = ("VOC_dataset",)
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = "X-101-32x8d.pkl"  # pre-trained backbone on ImageNet
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 200000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# train
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
print("training start")
trainer.train()

# inference with the trained model
cocoGt = COCO("test.json")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # the previously trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.DATASETS.TEST = ("VOC_dataset", )

predictor = DefaultPredictor(cfg)
coco_dt = []

print("start inference")
for imgid in cocoGt.imgs:
    file_name = cocoGt.loadImgs(ids=imgid)[0]['file_name']
    image = cv2.imread("test_images/" + file_name)[:, :, ::-1]  # load image
    out = predictor(image)  # run inference
    anno = out['instances'].to('cpu').get_fields()
    scores = anno['scores'].numpy()
    classes = anno['pred_classes'].numpy()
    masks = anno['pred_masks'].numpy()
    num_instance = len(scores)
    for i in range(num_instance):
        pred = {}
        pred['image_id'] = imgid
        pred['category_id'] = int(classes[i]) + 1
        pred['segmentation'] = binary_mask_to_rle(masks[i, :, :])
        pred['score'] = float(scores[i])
        coco_dt.append(pred)

with open("prediction.json", "w") as f:
    json.dump(coco_dt, f)
