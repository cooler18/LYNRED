import os
from os.path import join
import cv2
import lynred_py
from tqdm import tqdm

######################################################################################
import numpy as np

verbose = False
Folders = ["Day", "Night"]
Subfolders = ["master", "slave"]
p = "/home/godeta/PycharmProjects/LYNRED/Images"
pipe_master = lynred_py.algo.pipe_shutterless_2ref_t()
pipe_master_tone_mapping = lynred_py.algo.pipe_shutterless_2ref_t()
pipe_master_tone_mapping.load_from_file("/home/godeta/PycharmProjects/LYNRED/Images/Pipes/master/master_pipe_tone_mapping.spbin")
pipe_master.load_from_file("/home/godeta/PycharmProjects/LYNRED/Images/Pipes/master/master_pipe.spbin")
pipe_slave = lynred_py.algo.pipe_shutterless_2ref_t()
pipe_slave_tone_mapping = lynred_py.algo.pipe_shutterless_2ref_t()
pipe_slave_tone_mapping.load_from_file("/home/godeta/PycharmProjects/LYNRED/Images/Pipes/slave/slave_pipe_tone_mapping.spbin")
pipe_slave.load_from_file("/home/godeta/PycharmProjects/LYNRED/Images/Pipes/slave/slave_pipe.spbin")

count = 0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for f in Folders:
    for s in Subfolders:
        path_current = join(p, f, s)
        p_save = join(path_current, "infrared_corrected")
        if s == 'master':
            pipe = pipe_master
        else:
            pipe = pipe_slave
        print(f"\nFolder {path_current} in process...")
        for img in tqdm(os.listdir(join(path_current, "infrared"))):
            i = np.array(cv2.imread(join(path_current, "infrared", img), 0)).astype(np.uint16)*255
            i = lynred_py.base_t.image_t(i)
            i_proc = lynred_py.base_t.image_t()
            pipe.execute(i, i_proc)
            i_proc = np.array(i_proc)
            i = np.array(i)
            if i_proc.max()>255: #f == 'Day' or s == 'slave':
                i_proc = ((i_proc - i_proc.min())/(i_proc.max() - i_proc.min())*255).astype(np.uint8)
            else:
                i_proc = i_proc.astype(np.uint8)
            i_proc = clahe.apply(i_proc)
            # cv2.imshow('not processed image', i)
            # cv2.imshow('post processed image', i_proc)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(join(p_save, img), i_proc)
            count += 1
            # if count == 3:
            #     count = 0
            #     break

print(f"{count} images processed !")
