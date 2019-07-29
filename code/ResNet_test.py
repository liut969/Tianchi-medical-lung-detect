from keras.models import load_model
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
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


def _main():
    model = load_model('./saved_models/trained_weights_final_ResNet.h5')
    model.summary()
    predict_path = './cut_csv/lt_yolov3_014_cut_testA.csv'
    data_path = '../data/testA'
    predict_anns_all = pd.read_csv(predict_path)
    target_size = 64
    save_name = './lt_yolov3_014_testA_ResNet.csv'
    seriesuid, coordX, coordY, coordZ, class_label, probability = [], [], [], [], [], []

    file_ids = get_file_id(data_path)
    for current_id in tqdm(file_ids):
        current_file = os.path.join(data_path, current_id + '.mhd')
        ct, origin, spacing = load_itk(current_file)
        predict_ann_df = predict_anns_all.query('seriesuid == "%s"' % current_id).copy()

        for _, predict_ann in predict_ann_df.iterrows():
            pre_x, pre_y, pre_z, pre_w, pre_h = predict_ann.coordX, predict_ann.coordY, predict_ann.coordZ, predict_ann.diameterX, predict_ann.diameterY
            current_image = ct[int(pre_z)]

            w, h = int(pre_w), int(pre_h)
            pre_x_min, pre_x_max = int(pre_x - pre_w / 2), int(pre_x + pre_w / 2)
            pre_y_min, pre_y_max = int(pre_y - pre_h / 2), int(pre_y + pre_h / 2)
            if w > target_size or h > target_size:
                max_size = int(max(w, h))
                result_image = np.zeros((max_size, max_size))
                result_image[0:int(h), 0:int(w)] = current_image[pre_y_min:pre_y_max, pre_x_min:pre_x_max]
                result_image = cv2.resize(result_image, (target_size, target_size))
            else:
                result_image = current_image[pre_y_min:pre_y_max, pre_x_min:pre_x_max]
                result_image = cv2.copyMakeBorder(result_image,
                                                  int((target_size - h) / 2), math.ceil((target_size - h) / 2),
                                                  int((target_size - w) / 2), math.ceil((target_size - w) / 2),
                                                  cv2.BORDER_CONSTANT, value=0)
            result_image = normalize(result_image)

            predict_image = model.predict(result_image[np.newaxis, ::, ::, np.newaxis])
            # print(predict_image, current_id, pre_x, pre_y, pre_z)
            # fig, (ax0, ax1) = plt.subplots(1, 2)
            # ax0.imshow(current_image)
            # ax1.imshow(result_image)
            # plt.show()
            # if predict_image[0][1] + predict_ann.probability - predict_image[0][0] > 0:
            # if predict_image[0][1] - predict_image[0][0] > 0:
            if predict_image[0][1] + predict_ann.probability + 0.5 - predict_image[0][0] > 0:
                seriesuid.append(int(current_id))
                coordX.append(pre_x * spacing[2] + origin[2])
                coordY.append(pre_y * spacing[1] + origin[1])
                coordZ.append(pre_z * spacing[0] + origin[0])
                class_label.append(int(predict_ann.label))
                probability.append(predict_ann.probability)

    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ, 'class': class_label, 'probability': probability})
    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    dataframe.to_csv(save_name, index=False, sep=',', columns=columns)




if __name__ == '__main__':
    _main()
