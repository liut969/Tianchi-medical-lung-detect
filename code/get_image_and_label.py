
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

def normalize(image):
    MIN_BOUND = np.min(image)
    MAX_BOUND = np.max(image)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def get_file_id(file_path):
    files = []
    for f_name in [f for f in os.listdir(file_path) if f.endswith('.mhd')]:
        f_name = f_name.replace('.mhd', '')
        files.append(f_name)
    return sorted(files)

def load_itk(file):
    itkimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing

def get_image_and_label(sets=['train_part1'],
                        data_path='../data',
                        anns_path='../data/chestCT_round1_annotation.csv',
                        predict_path=['./cut_csv/lt_yolov3_014_cut_part1.csv'],
                        target_size=64,
                        ):
    images, labels = [], []
    anns_all = pd.read_csv(anns_path)

    for i, current_set in enumerate(sets):
        predict_anns_all = pd.read_csv(predict_path[i])
        from_path = os.path.join(data_path, current_set)
        file_ids = get_file_id(from_path)
        for current_id in tqdm(file_ids):
            current_file = os.path.join(from_path, current_id + '.mhd')
            ct, origin, spacing = load_itk(current_file)

            ann_df = anns_all.query('seriesuid == "%s"' % current_id).copy()
            ann_df.coordX = (ann_df.coordX - origin[2]) / spacing[2]
            ann_df.coordY = (ann_df.coordY - origin[1]) / spacing[1]
            ann_df.coordZ = (ann_df.coordZ - origin[0]) / spacing[0]
            ann_df.diameterX = ann_df.diameterX / spacing[2]
            ann_df.diameterY = ann_df.diameterY / spacing[1]
            ann_df.diameterZ = ann_df.diameterZ / spacing[0]
            predict_ann_df = predict_anns_all.query('seriesuid == "%s"' % current_id).copy()

            for _, predict_ann in predict_ann_df.iterrows():
                pre_x, pre_y, pre_z, pre_w, pre_h = predict_ann.coordX, predict_ann.coordY, predict_ann.coordZ, predict_ann.diameterX, predict_ann.diameterY
                flag = False
                for _, ann in ann_df.iterrows():
                    x, y, z, w, h, d = int(ann.coordX), int(ann.coordY), int(ann.coordZ), int(ann.diameterX), int(ann.diameterY), int(ann.diameterZ)
                    if pre_z > z - d / 2 and pre_z < z + d / 2:
                        x_min = (x - w / 2)
                        y_min = (y - h / 2)
                        x_max = (x + w / 2)
                        y_max = (y + h / 2)
                        if pre_x > x_min and pre_x < x_max and pre_y > y_min and pre_y < y_max:
                            flag = True
                            current_image = ct[int(pre_z)]
                            max_size = int(max(w, h))
                            result_image = np.zeros((max_size, max_size))
                            result_image[0:int(h), 0:int(w)] = current_image[int(y_min):int(y_max), int(x_min):int(x_max)]
                            result_image = cv2.resize(result_image, (target_size, target_size))
                            result_image = normalize(result_image)
                            images.append(result_image)
                            labels.append(flag)
                            break
                if flag == False:
                    current_image = ct[int(pre_z)]
                    w, h = int(pre_w), int(pre_h)
                    pre_x_min, pre_x_max = int(pre_x - pre_w / 2), int(pre_x + pre_w / 2)
                    pre_y_min, pre_y_max = int(pre_y - pre_h / 2), int(pre_y + pre_h / 2)
                    max_size = int(max(w, h))
                    result_image = np.zeros((max_size, max_size))
                    result_image[0:int(h), 0:int(w)] = current_image[pre_y_min:pre_y_max, pre_x_min:pre_x_max]
                    result_image = cv2.resize(result_image, (target_size, target_size))
                    result_image = normalize(result_image)
                    images.append(result_image)
                    labels.append(flag)

    images = np.asarray(images)
    labels = np.asarray(labels)
    return np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1)), np.reshape(labels, (labels.shape[0], 1))



if __name__ == '__main__':
    images, labels = get_image_and_label(sets=['train_part1'],
                                        data_path='../data',
                                        anns_path='../data/chestCT_round1_annotation.csv',
                                        predict_path=['./lt_yolov3_014_cut_part1.csv'],
                                        target_size=64,
                                        )
    print(images.shape)
    print(labels.shape)
    for i in range(images.shape[0]):
        fig, (ax0) = plt.subplots(1, 1)
        current_image = images[i]
        current_image = np.reshape(current_image, (current_image.shape[0], current_image.shape[1]))
        ax0.imshow(current_image)
        print(labels[i])
        plt.show()


