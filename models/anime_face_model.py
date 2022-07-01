import sys
import os
import os.path as osp
sys.path.append(os.getcwd())
from models.base_model import BaseCVServiceModel
from models.recognition.preprocess import get_x2_box
from models.recognition import get_rec_model

import torch
from mmdet.apis import init_detector, inference_detector
import time

import torch
import numpy as np
import cv2

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnimeFaceModel(BaseCVServiceModel):
    def __init__(self, path_prefix='/home/song/web_Projects/web_demo/crossd_web_demo/', device='cuda:0'):
        self.device = device
        self.path_prefix = path_prefix
        self.init_detector()
        self.init_recognizer()

    def init_detector(self):
        det_config = osp.join(self.path_prefix, 'checkpoints/detection/configs/icartoonface/icf_config.py')
        det_checkpoint = osp.join(self.path_prefix, 'checkpoints/detection/dcnv2_ep9.pth')
        self.detector = init_detector(det_config, det_checkpoint, self.device)

    def init_recognizer(self):
        rec_model_name = 'r50'
        rec_checkpoint = osp.join(self.path_prefix, 'checkpoints/recognition/129809.pth')
        self.recognizer = get_rec_model(rec_model_name, fp16=False)
        self.recognizer.load_state_dict(torch.load(rec_checkpoint))
        self.recognizer = self.recognizer.to(self.device)
        self.recognizer.eval()

    def preprocess4rec(self, img):
        img = cv2.resize(img, (256, 256))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().cuda()
        img.div_(255).sub_(0.5).div_(0.5)
        return img


    def thr_filter(self, result, thr):
        ret = []
        for box in result:
            if box[4] > thr:
                ret.append(box)
        return ret

    def get_box4rec(self, faces, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor_imgs = None
        for face in faces:
            coords = [int(coord) for coord in face[: 4]]
            img = get_x2_box(coords, image)
            img = self.preprocess4rec(img)
            if tensor_imgs is None:
                tensor_imgs = img
            else:
                tensor_imgs = torch.cat((tensor_imgs, img), dim=0)
        return tensor_imgs

    def run(self, path, **kwargs):
        mode = kwargs.get('mode', 'det')
        thr = kwargs.get('thr', 0.85)
        out_file = kwargs.get('out_file', None)
        if mode not in ['det', 'feat']:
            return f"Expected mode is 'feat' or 'det', got {mode}"
        t1 = time.time()  
        img = cv2.imread(path)
        result = inference_detector(self.detector, img)
        det_result = self.thr_filter(result[0].tolist(), thr)
        if len(det_result) == 0:
            det_result = [[ img.shape[1] // 4, img.shape[0] // 4, img.shape[1] * 3 // 4, img.shape[0] * 3 // 4, 0.99 ]] # if no face detected, return center part
        t2 = time.time()   
        if out_file is not None:
            self.detector.show_result(img, result, thr, out_file=out_file)
        if mode == 'det':
            logger.info(f"{len(det_result)} face(s) detected, det time(ms): {(t2-t1)*1000:.0f}")
            return det_result
        tensor_imgs  = self.get_box4rec(det_result, img)
        with torch.no_grad():
            feat = self.recognizer(tensor_imgs).cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        t3 = time.time()    
        logger.info(f"{len(det_result)} face(s) detected, det time(ms): {(t2-t1)*1000:.0f}, rec time(ms): {(t3-t2)*1000:.0f}")
        return feat.tolist()


    def post_run(self):
        pass

if __name__ == '__main__':
    model = AnimeFaceModel(os.getcwd())
    image_path = 'assets/test.jpg'
    res = model.run(image_path, mode='feat')
    print(res)


