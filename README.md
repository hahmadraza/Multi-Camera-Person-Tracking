## MULTI-CAMERA-PERSON-TRACKING

![alt text](https://github.com/hahmadraza/Multi-Camera-Person-Tracking/blob/main/resources/methadology.png)

### Clone and install dependencies
```bash
git clone https://github.com/hahmadraza/Multi-Camera-Person-Tracking.git
cd Multi-Camera-Person-Tracking
pip install -v -e .
```
### RUN code
Extract frames from videos from all cameras at /path/to/folder/

```bash
/path/to/folder/cam1_img1.jpg
/path/to/folder/cam1_img2.jpg
/path/to/folder/cam2_img1.jpg
/path/to/folder/cam2_img2.jpg
.
.
.
/path/to/folder/camn_img1.jpg
/path/to/folder/camn_img2.jpg
```
Run following command to perform object detection on all frames 
```bash
python examples/track.py --tracking-method deepocsort --source /path/to/folder/ --yolo-model yolov8n.pt --reid-model osnet_x1_0_msmt17.pt --classes 0 --imgsz 352 --save-txt --hide-label --hide-conf --name 'exp_name'
```
Replace yolo-model and imgsz with desired yolo varient and image size respectively. 

### PLot tracks
```bash
python examples/multi_track.py --frames_dir /path/to/frames/dir --labels_dir /path_to_labels_dir
```