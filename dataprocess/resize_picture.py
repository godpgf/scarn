import cv2
import os
import re


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
    # from_path = sys.argv[1]
    # to_path = sys.argv[2]
    # from_path = "D:\\test_sr\\img"
    # to_path = "D:\\test_sr\\input"

    # from_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_HR"
    # to_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X2"
    # last_name = "x2"
    # scale = 1.0 / 2.0

    # from_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X2"
    # to_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X3"
    # last_name = "x3"
    # scale = 2.0 / 3.0

    # from_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X2"
    # to_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X4"
    # last_name = "x4"
    # scale = 2.0 / 4.0

    # from_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X3"
    # to_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X5"
    # last_name = "x5"
    # scale = 3.0 / 5.0

    from_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X3"
    to_path = "D:\lenovo_proj\PycharmProjects\scarn\dataset\DIV2K\DIV2K_train_LR_bicubic\X6"
    last_name = "x6"
    scale = 3.0 / 6.0

    inputs = get_input_list(from_path)
    for idx, input_path in enumerate(inputs):
        print("Processing {}".format(input_path))
        im_input = cv2.imread(input_path, -1)  # -1 means read as is, no conversions.
        if im_input.shape[2] == 4:
            print("Input {} has 4 channels, dropping alpha".format(input_path))
            im_input = im_input[:, :, :3]
        h, w, c = im_input.shape
        im_output = cv2.resize(im_input, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
        fname = os.path.splitext(os.path.basename(input_path))[0]
        if fname[-2] == "x":
            fname = fname[:-2]
        cv2.imwrite(os.path.join(to_path, fname + last_name + ".png"), im_output)
