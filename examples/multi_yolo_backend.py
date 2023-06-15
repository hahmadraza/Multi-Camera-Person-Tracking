from pathlib import Path
import numpy as np
import torch

from boxmot.utils.checks import TestRequirements
tr = TestRequirements()

from ultralytics.yolo.engine.results import Boxes, Results
from boxmot.utils import logger as LOGGER


class MultiYolo():
    def __init__(self, model, device, args):
        self.args = args
        self.device = device
        
        self.model_name = str(model.stem).lower()

        if 'yolov8' in self.model_name:
            self.model = model
            
    def try_sg_import(self):
        try:
            import super_gradients  # for linear_assignment
        except (ImportError, AssertionError, AttributeError):
            tr.check_packages(('super-gradients==3.1.1',))  # install
            
    def __call__(self, im, im0s):
        if 'yolov8' in self.model_name:
            preds = self.model(
                im,
                augment=False,
                visualize=False
            )
        else:
            LOGGER.error('The Yolo model you selected is not available')
            exit()
        return preds
    
    def overwrite_results(self, i, im0_shape, predictor):
        # overwrite bbox results with tracker predictions
        if predictor.tracker_outputs[i].size != 0:
            predictor.results[i].boxes = Boxes(
                # xyxy, (track_id), conf, cls
                boxes=torch.from_numpy(predictor.tracker_outputs[i]).to(self.device),
                orig_shape=im0_shape,  # (height, width)
            )
    
    def filter_results(self, i, predictor):
        if predictor.tracker_outputs[i].size != 0:
            # filter boxes masks and pose results by tracking results
            predictor.tracker_outputs[i] = predictor.tracker_outputs[i][predictor.tracker_outputs[i][:, 5].argsort()[::-1]]
            yolo_confs = predictor.results[i].boxes.conf.cpu().numpy()
            tracker_confs = predictor.tracker_outputs[i][:, 5]
            mask = np.in1d(yolo_confs, tracker_confs)
            
            if predictor.results[i].masks is not None:
                predictor.results[i].masks = predictor.results[i].masks[mask]
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
            elif predictor.results[i].keypoints is not None:
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
                predictor.results[i].keypoints = predictor.results[i].keypoints[mask]

    def postprocess(self, path, preds, im ,im0s, predictor):
        if 'yolo_nas' in self.model_name or 'yolox' in self.model_name:
            predictor.results[0] = Results(
                path=path,
                boxes=preds,
                orig_img=im0s[0],
                names=predictor.model.names
            )
        else:
            predictor.results = predictor.postprocess(preds, im, im0s)
        return predictor.results

        


if __name__ == "__main__":
    yolo = MultiYolo(model='YOLO_NAS_S', device='cuda:0')
    rgb = np.random.randint(255, size=(640, 640, 3),dtype=np.uint8)
    yolo(rgb, rgb)
            
