import cv2
import json
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO
from itertools import groupby
from pycocotools import mask as maskutil


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


model_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

# config
cocoGt = COCO("test.json")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # the trained model
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
