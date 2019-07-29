import os
import numpy as np
import SimpleITK as sitk
import pandas as pd

from tqdm import tqdm

label_dict = {}
label_dict[1] = 0
label_dict[5] = 1
label_dict[31] = 2
label_dict[32] = 3

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

    sets = ['train_part1', 'train_part2', 'train_part3', 'train_part4', 'train_part5']
    # sets = ['train_part1']
    data_path = '../data'
    png_image_path = './image_png'
    list_file = open('train.txt', 'w')
    anns_path = '../data/chestCT_round1_annotation.csv'

    anns_all = pd.read_csv(anns_path)
    anns_all['label'] = anns_all.label.apply(lambda x: label_dict[x])
    for i, current_set in enumerate(sets):
        from_path = os.path.join(data_path, current_set)
        file_ids = get_file_id(from_path)
        for seriesuid in tqdm(file_ids):
            current_file = os.path.join(from_path, seriesuid + '.mhd')
            ct, origin, spacing = load_itk(current_file)

            ann_df = anns_all.query('seriesuid == "%s"' % seriesuid).copy()
            ann_df.coordX = (ann_df.coordX - origin[2]) / spacing[2]
            ann_df.coordY = (ann_df.coordY - origin[1]) / spacing[1]
            ann_df.coordZ = (ann_df.coordZ - origin[0]) / spacing[0]
            ann_df.diameterX = ann_df.diameterX / spacing[2]
            ann_df.diameterY = ann_df.diameterY / spacing[1]
            ann_df.diameterZ = ann_df.diameterZ / spacing[0]

########### 全部图像
            # for num in range(ct.shape[0]):
            #     png_image_dir = os.path.join(png_image_path, 'image_png_part' + str(i+1))
            #     save_name = os.path.join(png_image_dir, seriesuid + '_' + str(num).zfill(3) + '.png')
            #     list_file.write(save_name)
            #     for _, ann in ann_df.iterrows():
            #         x, y, z, w, h, d = ann.coordX, ann.coordY, ann.coordZ, ann.diameterX, ann.diameterY, ann.diameterZ
            #         if num > z - d/2 and num < z + d / 2:
            #             x_min = int(x - w / 2)
            #             y_min = int(y - h / 2)
            #             x_max = int(x + w / 2)
            #             y_max = int(y + h / 2)
            #             # print(x_min, y_min, x_max, y_max, int(ann.label))
            #             list_file.write(' ' + str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + ',' + str(int(ann.label)))
            #     list_file.write('\n')

############ 只含有病变区域

            for num in range(ct.shape[0]):
                location = []
                for _, ann in ann_df.iterrows():
                    x, y, z, w, h, d = ann.coordX, ann.coordY, ann.coordZ, ann.diameterX, ann.diameterY, ann.diameterZ
                    if num > z - d/2 and num < z + d / 2:
                        x_min = int(x - w / 2)
                        y_min = int(y - h / 2)
                        x_max = int(x + w / 2)
                        y_max = int(y + h / 2)
                        location.append([x_min, y_min, x_max, y_max, ann.label])
                if not len(location) == 0:
                    save_name = os.path.join(png_image_path, seriesuid + '_' + str(num).zfill(3) + '.png')
                    list_file.write(save_name)
                    for current_location in location:
                        list_file.write(' '
                                        + str(current_location[0]) + ',' + str(current_location[1]) + ','
                                        + str(current_location[2]) + ',' + str(current_location[3]) + ','
                                        + str(int(current_location[4])))
                    list_file.write('\n')



if __name__ == '__main__':
    _main()
