from pathlib import Path
import yaml
from types import SimpleNamespace
from boxmot.utils import BOXMOT


def get_tracker_config(tracker_type):
    tracking_config = \
        BOXMOT /\
        tracker_type /\
        'configs' /\
        (tracker_type + '.yaml')
    return tracking_config
    

def create_tracker(tracker_type, tracker_config, reid_weights, device, half):

    with open(tracker_config, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg = SimpleNamespace(**cfg)  # easier dict acces by dot, instead of ['']
        
    if tracker_type == 'deepocsort':
        from boxmot.deepocsort.ocsort import OCSort
        deepocsort = OCSort(
            reid_weights,
            device,
            half,
            det_thresh=cfg.det_thresh,
            max_age=cfg.max_age,
            min_hits=cfg.min_hits,
            iou_threshold=cfg.iou_thresh,
            delta_t=cfg.delta_t,
            asso_func=cfg.asso_func,
            inertia=cfg.inertia,
        )
        return deepocsort
    else:
        print('No such tracker')
        exit()