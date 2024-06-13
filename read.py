import pydicom
import matplotlib.pyplot as plt
import numpy as np
from pydicom.data import get_testdata_files
import cv2

def read_dcm_image(path):
    ds = pydicom.dcmread(path)
    print("read from:" + path)
    print(ds.pixel_array.shape)
    plt.imsave("test.png", ds.pixel_array, cmap=plt.cm.gray)
    return "test.png"



if __name__ == "__main__":
    dicom_path = get_testdata_files("CT_small.dcm")[0]
    image_path = read_dcm_image(dicom_path)
    cv2.imshow("test", cv2.imread(image_path))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

