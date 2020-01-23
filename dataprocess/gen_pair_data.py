import cv2
import os
import sys
import re
import numpy as np


def get_input_list(path):
    regex = re.compile(".*.(png|jpeg|jpg|tif|tiff)")
    if os.path.isdir(path):
        inputs = os.listdir(path)
        inputs = [os.path.join(path, f) for f in inputs if regex.match(f)]
        # log.info("Directory input {}, with {} images".format(path, len(inputs)))

    elif os.path.splitext(path)[-1] == ".txt":
        dirname = os.path.dirname(path)
        with open(path, 'r') as fid:
            inputs = [l.strip() for l in fid.readlines()]
        inputs = [os.path.join(dirname, 'input', im) for im in inputs]
        # log.info("Filelist input {}, with {} images".format(path, len(inputs)))
    elif regex.match(path):
        inputs = [path]

    return inputs


if __name__ == "__main__":
    img_path = sys.argv[1]
    from_path = sys.argv[2]
    to_path = sys.argv[3]
    # img_path = "D:\\2k\\full\\DIV2K_valid_HR"
    # from_path = "D:\\2k\\DIV2K\\DIV2K_valid_HR"
    # to_path = "D:\\2k\\DIV2K\\DIV2K_valid_LR_bicubic"
    inputs = get_input_list(img_path)
    for idx, input_path in enumerate(inputs):
        print("Processing {}".format(input_path))
        im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.
        if im_input.shape[2] == 4:
            print("Input {} has 4 channels, dropping alpha".format(input_path))
            im_input = im_input[:, :, :3]

        img_gray = cv2.cvtColor(im_input, cv2.COLOR_BGR2GRAY)
        # im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
        fname = os.path.splitext(os.path.basename(input_path))[0]
        cv2.imwrite(os.path.join(from_path, fname + ".png"), img_gray)
        x, y = img_gray.shape[0:2]
        low_img_gray_x2 = cv2.resize(img_gray, (y//2, x//2), interpolation=cv2.INTER_CUBIC)
        low_img_gray_x4 = cv2.resize(low_img_gray_x2, (y//4, x//4), interpolation=cv2.INTER_CUBIC)
        blur_img_gray_x2 = cv2.resize(low_img_gray_x2, (y, x))
        blur_img_gray_x4 = cv2.resize(low_img_gray_x4, (y, x))
        cv2.imwrite(os.path.join(to_path, "X2", fname + "x2.png"), blur_img_gray_x2)
        cv2.imwrite(os.path.join(to_path, "X4", fname + "x4.png"), blur_img_gray_x4)
