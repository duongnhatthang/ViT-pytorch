import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import sys
import pydicom
from pydicom.data import get_testdata_files

selected_wave = "BL" #BL, M12, M24, M36

def get_paths():
    path_list = glob.glob("/Volumes/Elements/MOAKS/*")
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
    label_names = []
    out_path = []
    out_labels = []
    for i, id in enumerate(id_list):
        if wave[i] != selected_wave:
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


df = pd.read_csv("paths_and_labels.csv")
path_list = list(df['path'])
es_list = list(df['es'])

# min_value = 200

for p in path_list:
    slice_path_list = glob.glob(p+'/'+selected_wave+'/AXLD'+'/*')
    min_value = min(min_value, len(slice_path_list))

    # count = 0
    # for slice_path in slice_path_list:
    #     read_dicom(slice_path)
    #     break

    #     s = pydicom.dcmread(slice_path)
    #     loc = s.get('SliceLocation')
    #     if loc is not None:
    #         count+=1
    #         print(loc)
    # if count > 0:
    #     print(f"{p}, count = {count}")
    #     count = 0

    # break
print(f"min_value={min_value}")