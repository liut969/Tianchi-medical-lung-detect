# [天池比赛-肺部CT多病种智能诊断-全球数据智能大赛(2019)——“数字人体”赛场一](https://tianchi.aliyun.com/competition/entrance/231724/introduction)
## 说明
赛题数据说明，[请看链接](https://tianchi.aliyun.com/competition/entrance/231724/information)。

本次医疗比赛采用YOLOv3进行病灶检测，使用ResNet进行假阳性衰减，实现过程<font color="#dd0000">简单，直接</font>，准确率一般，但适合新人上手，可以作为baseline。
## 代码目录说明
```
|--data
    |--testA
    |--train_part1
    |--train_part2
    |--train_part3
    |--train_part4
    |--train_part5
    |--chestCT_round1_annotation.csv
|--code
    |--kmeans-anchor-boxes-master
    |--model_data
    |--yolo3
    ...
    |--yolov3.weights
|--README.md
```
## 代码使用说明
- [下载赛题数据](https://tianchi.aliyun.com/competition/entrance/231724/information)，将数据放在```data```文件夹下。
- [下载YOLOv3权重 yolov3.weights](https://pjreddie.com/darknet/yolo/)，放在```code```文件夹下。
- 运行```python convert.py -w yolov3.cfg yolov3.weights ./yolo.h5```，生成keras预训练权重。
- 运行```generate_the_image.py```生成待训练图片。
- 运行```lt_annotation.py```生成注释文件train.txt。
- 运行```lt_train.py```进行YOLO训练。
- 运行```lt_yolo_image.py```生成预测结果。
- 运行```ResNet_train.py```进行假阳性网络训练。
- 运行```ResNet_test.py```生成最终结果csv文件。
## 其他说明
- 环境和硬件：Python 3.5.4，Keras 2.2.4，GPU 1080Ti。
- 如使用自己的锚，请运行```code/kmeans-anchor-boxes-master/lt_get_anchor.py```，将生成的锚放在```code/model_data/lt_yolo_anchors.txt```中。