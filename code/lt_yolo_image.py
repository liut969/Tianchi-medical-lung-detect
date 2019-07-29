import argparse
from lt_yolo import YOLO
from PIL import Image
import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

label_dict = {}
label_dict['nodule'] = 1
label_dict['stripe'] = 5
label_dict['artery'] = 31
label_dict['lymph'] = 32

def get_file_name(file_path):
    files = []
    for f_name in [f for f in os.listdir(file_path) if f.endswith('.mhd')]:
        files.append(f_name)
    return sorted(files)

def load_itk(file):
    itkimage = sitk.ReadImage(file)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def detect_img_lt(yolo, clipmin=-1000, clipmax=600):
    file_path = '../data/testA'
    save_name = './lt_yolov3_014.csv'
    seriesuid, coordX, coordY, coordZ, class_label, probability = [], [], [], [], [], []

    files = get_file_name(file_path)
    for f_name in tqdm(files):
        result_id = int(f_name.replace('.mhd', ''))
        current_file = os.path.join(file_path, f_name)
        ct, origin, spacing = load_itk(current_file)
        ct = ct.clip(min=clipmin, max=clipmax)
        for num in range(ct.shape[0]):
            image = Image.fromarray(ct[num])
            detect_result = yolo.detect_image(image)
            for one_result in detect_result:
                result_probability = one_result[1]
                result_label = int(label_dict[one_result[0]])
                result_x = (one_result[2] + one_result[4]) / 2
                result_x = result_x * spacing[2] + origin[2]
                result_y = (one_result[3] + one_result[5]) / 2
                result_y = result_y * spacing[1] + origin[1]
                result_z = num
                result_z = result_z * spacing[0] + origin[0]
                # print(result_id, result_x, result_y, result_z, result_label, result_probability)
                seriesuid.append(result_id)
                coordX.append(result_x)
                coordY.append(result_y)
                coordZ.append(result_z)
                class_label.append(result_label)
                probability.append(result_probability)
    dataframe = pd.DataFrame({'seriesuid': seriesuid, 'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ, 'class': class_label, 'probability': probability})
    columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'probability']
    dataframe.to_csv(save_name, index=False, sep=',', columns=columns)
    yolo.close_session()


def detect_img_lt_like_annotation(yolo, clipmin=-1000, clipmax=600):
    file_paths = ['../data/train_part1', '../data/train_part2', '../data/train_part3', '../data/train_part4', '../data/train_part5', '../data/testA']
    save_names = ['./cut_csv/lt_yolov3_014_cut_part1.csv', './cut_csv/lt_yolov3_014_cut_part2.csv',
                  './cut_csv/lt_yolov3_014_cut_part3.csv', './cut_csv/lt_yolov3_014_cut_part4.csv',
                  './cut_csv/lt_yolov3_014_cut_part5.csv', './cut_csv/lt_yolov3_014_cut_testA.csv']

    if not os.path.isdir('./cut_csv'):
        os.mkdir('./cut_csv')
    for i in range(len(file_paths)):
        file_path = file_paths[i]
        save_name = save_names[i]
        seriesuid, coordX, coordY, coordZ, diameterX, diameterY, diameterZ, label, probability = [], [], [], [], [], [], [], [], []

        files = get_file_name(file_path)
        for f_name in tqdm(files):
            result_id = int(f_name.replace('.mhd', ''))
            current_file = os.path.join(file_path, f_name)
            ct, origin, spacing = load_itk(current_file)
            ct = ct.clip(min=clipmin, max=clipmax)
            for num in range(ct.shape[0]):
                image = Image.fromarray(ct[num])
                detect_result = yolo.detect_image(image)
                for one_result in detect_result:
                    result_probability = one_result[1]
                    result_label = int(label_dict[one_result[0]])
                    seriesuid.append(result_id)
                    label.append(result_label)
                    probability.append(result_probability)
                    coordX.append((one_result[4] + one_result[2]) / 2)
                    coordY.append((one_result[3] + one_result[5]) / 2)
                    coordZ.append(num)
                    diameterX.append(one_result[4] - one_result[2])
                    diameterY.append(one_result[5] - one_result[3])
                    diameterZ.append(1)
        dataframe = pd.DataFrame({'seriesuid': seriesuid, 'coordX': coordX, 'coordY': coordY, 'coordZ': coordZ,
                                  'diameterX': diameterX, 'diameterY': diameterY, 'diameterZ':diameterZ,
                                  'label': label, 'probability': probability})
        columns = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameterX', 'diameterY', 'diameterZ', 'label', 'probability']
        dataframe.to_csv(save_name, index=False, sep=',', columns=columns)
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image',
        default=True,
        # dest='flag',
        # action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False,default='',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()
    print(FLAGS)

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        # detect_img_lt(YOLO(**vars(FLAGS)))
        detect_img_lt_like_annotation(YOLO(**vars(FLAGS)))
