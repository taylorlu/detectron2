# detectron2 for [PubLayNet](https://arxiv.org/abs/1908.07836)
* Document layout analysis using detectron2 framework, this is the [repos of PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)
 
## Installation
### build detectron2
1. `python3 setup.py install`

## Usage
1. document layout analysis
* `python3 demo/demo.py --config-file configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml --input test.png  --output ./ --confidence-threshold 0.5 --opts MODEL.WEIGHTS model_final_trimmed.pth MODEL.DEVICE cpu`

<table>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/layout1.png" height="720" width="640" ></th>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/layout2.png" height="720" width="640" ></th>
  </tr>
  <tr><th>  </th></tr>
  <tr><th>  </th></tr>
  <tr> 
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/layout3.png" height="720" width="640" ></th>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/layout4.png" height="720" width="640" ></th>
  </tr>
  <tr><th>  </th></tr>
  <tr><th>  </th></tr>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/layout5.png" height="720" width="640" ></th>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/layout6.png" height="720" width="640" ></th>
  </tr>
</table>


2. object detection by faster-rcnn
* `python3 demo/demo.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --input test.jpg --output ./ --opts MODEL.WEIGHTS model_final_faster_rcnn.pkl MODEL.DEVICE cpu`

<table>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/detect1.jpg" height="640" width="960" ></th>
  </tr>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/detect2.jpg" height="640" width="960" ></th>
  </tr>
</table>


3. object detection by mask-rcnn
* `python3 demo/demo.py --config-file configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --input test.jpg --output ./ --opts MODEL.WEIGHTS model_final_mask_rcnn.pkl MODEL.DEVICE cpu`

<table>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/mask1.jpg" height="640" width="960" ></th>
  </tr>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/mask2.jpg" height="640" width="960" ></th>
  </tr>
</table>


4. human joint keypoint detection
* `python3 demo/demo.py --config-file configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --input test.jpg --output ./ --opts MODEL.WEIGHTS model_final_keypoint_rcnn.pkl MODEL.DEVICE cpu`

<table>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/keypoint1.jpg" height="640" width="960" ></th>
  </tr>
  <tr>
    <th><img src="https://github.com/taylorlu/detectron2/blob/master/imgs/keypoint2.jpg" height="640" width="960" ></th>
  </tr>
</table>

## Reference
* more models can be found in [MODEL_ZOO](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)
