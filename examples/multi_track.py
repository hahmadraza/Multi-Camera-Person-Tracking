import argparse
from pathlib import Path
import copy
import cv2
import operator
import os
import time
import torch
import numpy as np
from pathlib import Path
from torch.nn import functional as F
from __future__ import division, print_function, absolute_import


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness, text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
    cv2.putText(
        frame, str(track_id), (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=text_thickness)

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness

def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size()  # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input1.dim()
    )
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(
        input2.dim()
    )
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat

def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = mat1 + mat2
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat

def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat

def compute_distance(qf, gf, dist_metric):
    distmat = compute_distance_matrix(qf, gf, dist_metric)
    # print(distmat.shape)
    return distmat.numpy()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=Path, help='/path/to/raw-frames-folder')
    parser.add_argument('--labels_dir', type=Path, help='/path/to/predicted/labels/from/object/detector')
    parser.add_argument('--dist_metric', type=str, default='euclidean', help='distance metric to be used')
    parser.add_argument('--threshold', type=float, default=750, help='Threshold for euclidean distance')

    args = parser.parse_args()
    return args    

def main(args):
    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []

    frames_dir = args.frames_dir
    labels_dir = args.labels_dir

    all_frames = os.listdir(frames_dir)
    all_frames.sort()
    for frame in all_frames:
        im = cv2.imread(os.path.join(frames_dir, frame))
        img_h, img_w = im.shape[0], im.shape[1]
        track_file = os.path.join(labels_dir, frame.split('.')[0]+'.txt')
        if not os.path.isfile(track_file):
            continue
        tracks = open(track_file, "r").readlines()
        tmp_ids = []
        for line in tracks:
            values = line.split()

            # Extract relevant information from the line
            label = values[0]
            x = float(values[1])
            y = float(values[2])
            width = float(values[3])
            height = float(values[4])
            id = int(values[5])
            x1, y1 = x-width/2, y-height/2
            x2, y2 = x+width/2, y+height/2
            x1, y1 = int(x1*img_w), int(y1*img_h)
            x2, y2 = int(x2*img_w), int(y2*img_h)

            area = (x2 - x1) * (y2 - y1)
            if id not in track_cnt:
                track_cnt[id] = [
                    [frame.split('.')[0], x1, y1, x2, y2, area]
                ]
                images_by_id[id] = [im[y1:y2, x1:x2]]
            else:
                track_cnt[id].append([
                    frame.split('.')[0], x1, y1, x2, y2, area
                ])
                images_by_id[id].append(im[y1:y2, x1:x2])

                tmp_ids.append(id)
        ids_per_frame.append(set(tmp_ids))

    device = 'cpu'
    fp16 = False

    from boxmot.deep.reid_multibackend import ReIDDetectMultiBackend
    embedder = ReIDDetectMultiBackend(weights=Path('osnet_x1_0_msmt17.pt'), device=device, fp16=fp16)
    
    t1 = time.time()
    # reid = REID()

    print(f'Total IDs = {len(images_by_id)}')
    feats = dict()

    for i in images_by_id:
        print(f'ID number {i} -> Number of frames {len(images_by_id[i])}')
        # feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])
        with torch.no_grad():
            to_process = min(len(images_by_id[i]),400)
            print('Processing {} crops'.format(to_process))
            feats[i] = embedder(images_by_id[i][:to_process])

    exist_ids = set()
    final_fuse_id = dict()

    for f in copy.deepcopy(ids_per_frame):
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids or f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    if len(images_by_id[nid]) < 10:
                        exist_ids.add(nid)
                        continue
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        tmp = np.mean(compute_distance(feats[nid], feats[oid], args.dist_metric))
                        print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < args.threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]

    print('Final ids and their sub-ids:', final_fuse_id)
    print('MOT took {} seconds'.format(int(time.time() - t1)))
    t2 = time.time()
    # Generate a single video with complete MOT/ReID

    path = Path(labels_dir).parent.absolute()
    complete_path = os.path.join(path, 'complete3.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    temp_frame = cv2.imread(os.path.join(frames_dir, all_frames[0]))
    h,w = temp_frame.shape[0], temp_frame.shape[1]
    out = cv2.VideoWriter(complete_path, fourcc, 30, (w, h))

    for frame in all_frames:
        frame_path = os.path.join(frames_dir, frame)
        frame2 = cv2.imread(frame_path)

        for idx in final_fuse_id:
            for i in final_fuse_id[idx]:
                for f in track_cnt[i]:
                    # print('frame {} f0 {}'.format(frame,f[0]))
                    if frame.split('.')[0] == f[0]:
                        text_scale, text_thickness, line_thickness = get_FrameLabels(frame2)
                        cv2_addBox(idx, frame2, f[1], f[2], f[3], f[4], line_thickness, text_thickness, text_scale)
        out.write(frame2)
    out.release()
    print('Saved video to {}'.format(complete_path))





if __name__ == "__main__":
    args = parse_opt()
    print(args)
    main(args)

    
