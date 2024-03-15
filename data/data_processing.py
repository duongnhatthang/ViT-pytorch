import glob
import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import sys
import pydicom
from pydicom.data import get_testdata_files

selected_wave = "BL" #BL, M12, M24, M36

def save_paths_to_csv(input_path):
    path_list = glob.glob(input_path)
    names = [p.split("/")[-1] for p in path_list]
    # print(len(names))
    # print(names[:10])

    df = pd.read_csv("MOAKS20180911.csv")
    study = df['study']
    id = df['id']
    knee = df['knee']
    wave = df['wave']
    es = df['effusionsynovitis']

    id_list = list(id)
    knee_list = list(knee)
    es_list = list(es)
    study_list = list(study)
    label_names = []
    out_path = []
    out_labels = []
    for i, id in enumerate(id_list):
        if not isinstance(study_list[i], str) or wave[i] != selected_wave:
            continue
        name = str(id)+str(knee_list[i])
        label_names.append(name)
        if name in names:
            path_idx = names.index(name)
            out_path.append(path_list[path_idx])
            out_labels.append(es_list[i])
    out_df = {'path': out_path, 'es': out_labels}
    out_df = pd.DataFrame(data = out_df)
    out_df.to_csv("paths_and_labels.csv")
    # print(len(out_path))
    # print(len(out_labels))
    # print(out_path[:10])
    # print(out_labels[:10])

# print(__doc__)
#filename = get_testdata_files("bmode.dcm")[0]

def read_dicom(path):
    dataset = pydicom.dcmread(path)
    # Print all tags, for debugging
    for element in dataset:
        print(element)
    print('-----------------------------------')

    filename = path.split("/")[-1]
    # Normal mode:
    print()
    print("Filename.........:", filename)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


    # use .get() if not sure the item exists, and want a default value if missing
    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

    # plot the image using matplotlib
    # plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    # plt.show()

def processing_data_from_csv_paths(csv_path):
    df = pd.read_csv(csv_path)
    path_list = list(df['path'])
    es_list = list(df['es'])

    min_num_slices = 59
    num_continuous_slices = 20
    cache = np.zeros((len(path_list), min_num_slices, 384, 384))
    for i, p in enumerate(tqdm(path_list)):
        slice_path_list = glob.glob(p+'/'+selected_wave+'/AXLD'+'/*')

        for j, slice_path in enumerate(slice_path_list):
        #     read_dicom(slice_path)
        #     break
            if j>=min_num_slices:
                break
            s = pydicom.dcmread(slice_path)
            assert 'PixelData' in s, "PixelData not in s"
            # print(np.max(s.pixel_array), np.min(s.pixel_array), np.shape(s.pixel_array))
            # break
            cache[i,j] = s.pixel_array
    out = np.sum(cache, axis=-1)
    out = np.sum(out, axis=-1) #(N, min_num_slices)
    out = np.sum(out, axis=0) #(min_num_slices)
    min = np.sum(out)
    index = -1
    for i in tqdm(range(min_num_slices-num_continuous_slices)):
        cur_sum = np.sum(out[i:i+num_continuous_slices])
        if cur_sum < min:
            min = cur_sum
            index = i
    print(f"Choosing slides: {index}:{index+num_continuous_slices}")

# save_paths_to_csv("/Volumes/Elements/MOAKS/*")
processing_data_from_csv_paths("paths_and_labels.csv")