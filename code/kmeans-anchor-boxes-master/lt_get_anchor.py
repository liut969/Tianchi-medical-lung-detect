'''
reference:https://github.com/lars76/kmeans-anchor-boxes
'''
import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from kmeans import kmeans, avg_iou

CLUSTERS = 9

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
    data_path = '../../data'
    anns_path = '../../data/chestCT_round1_annotation.csv'
    dataset = []

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

            for num in range(ct.shape[0]):
                for _, ann in ann_df.iterrows():
                    x, y, z, w, h, d = ann.coordX, ann.coordY, ann.coordZ, ann.diameterX, ann.diameterY, ann.diameterZ
                    if num > z - d/2 and num < z + d / 2:
                        dataset.append([w, h])

    print(len(dataset))
    dataset = np.asarray(dataset)
    out = kmeans(dataset, k=CLUSTERS)
    print("Accuracy: {:.2f}%".format(avg_iou(dataset, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))



if __name__ == '__main__':
    _main()
