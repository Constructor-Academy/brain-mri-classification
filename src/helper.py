import pydicom as dicom
import os
import cv2
import pandas as pd


def find_dcm_images(dcm_folder):
    """
    Finds all .dcm files in the given folder and all subfolders
    and returns their absolute paths in an array
    """
    paths = []
    for root, dirs, files in os.walk(dcm_folder):
        for file in files:
            if file.endswith('.dcm'):
                #print(os.path.join(root, file))
                paths.append(os.path.join(root, file))
    return paths


def dcm2ndarray(image_path):
    ds = dicom.dcmread(image_path)
    # pixel_array is numpy ndarray
    return ds.pixel_array

    
def dcm2png(source_path, dest_path):
    pixel_array = dcm2ndarray(source_path)
    cv2.imwrite(dest_path, pixel_array)



