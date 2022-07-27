import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import read_all_lines, get_transform_ir, get_transform_vis


class LynredDataset_processed(Dataset):
    def __init__(self, list_filename, training, Time='Night'):
        if Time == 'Day':
            self.datapath_ir = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/infrared_projected'
            self.datapath_vis = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/right'
        else:
            self.datapath_ir = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/infrared_projected'
            self.datapath_vis = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/right'
        self.ir_filenames, self.visible_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.visible_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        ir_images = [x[0] for x in splits]
        visible_images = [x[1] for x in splits]
        return ir_images, visible_images

    def load_infrared(self, filename):
        return Image.open(filename)

    def load_visible(self, filename):
        return Image.open(filename).convert('RGB')

    def __len__(self):
        return len(self.ir_filenames)

    def __getitem__(self, index):
        ir_img = self.load_infrared(os.path.join(self.datapath_ir, self.ir_filenames[index]))
        vis_img = self.load_visible(os.path.join(self.datapath_vis, self.visible_filenames[index]))

        if self.training:
            w, h = ir_img.size
            crop_w, crop_h = 480, 360

            x1 = random.randint(0, w - crop_w)
            if random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            # random crop
            ir_img = ir_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            vis_img = vis_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

            # to tensor, normalize
            processed_ir = get_transform_ir()
            processed_vis = get_transform_vis()
            ir_img = processed_ir(ir_img)
            vis_img = processed_vis(vis_img)

            return {"infrared": ir_img,
                    "visible": vis_img}

        else:
            w, h = ir_img.size

            # normalize
            processed_ir = get_transform_ir()
            processed_vis = get_transform_vis()
            ir_img = processed_ir(ir_img).numpy()
            vis_img = processed_vis(vis_img).numpy()

            # pad to size 640x480
            top_pad = 480 - h
            right_pad = 640 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(ir_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(vis_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            return {"left": left_img,
                    "right": right_img,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "ir_filenames": self.ir_filenames[index],
                    "visible_filenames": self.visible_filenames[index]}


class LynredDataset(Dataset):
    def __init__(self, list_filename, training, Time='Night'):
        if Time == 'Day':
            self.datapath_ir = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/left'
            self.datapath_vis = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/right'
        else:
            self.datapath_ir = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Night/hybrid/left'
            self.datapath_vis = '/home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/hybrid/right'
        self.ir_filenames, self.visible_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.visible_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        ir_images = [x[0] for x in splits]
        visible_images = [x[1] for x in splits]
        return ir_images, visible_images

    def load_infrared(self, filename):
        return Image.open(filename)

    def load_visible(self, filename):
        return Image.open(filename).convert('RGB')

    def __len__(self):
        return len(self.ir_filenames)

    def __getitem__(self, index):
        ir_img = self.load_infrared(os.path.join(self.datapath_ir, self.ir_filenames[index]))
        vis_img = self.load_visible(os.path.join(self.datapath_vis, self.visible_filenames[index]))

        if self.training:
            w, h = ir_img.size
            crop_w, crop_h = 480, 360

            x1 = random.randint(0, w - crop_w)
            if random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.3 * h), h - crop_h)

            # random crop
            ir_img = ir_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            vis_img = vis_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))

            # to tensor, normalize
            processed_ir = get_transform_ir()
            processed_vis = get_transform_vis()
            ir_img = processed_ir(ir_img)
            vis_img = processed_vis(vis_img)

            return {"infrared": ir_img,
                    "visible": vis_img}

        else:
            w, h = ir_img.size

            # normalize
            processed_ir = get_transform_ir()
            processed_vis = get_transform_vis()
            ir_img = processed_ir(ir_img).numpy()
            vis_img = processed_vis(vis_img).numpy()

            # pad to size 640x480
            top_pad = 480 - h
            right_pad = 640 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(ir_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(vis_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            return {"left": left_img,
                    "right": right_img,
                    "top_pad": top_pad,
                    "right_pad": right_pad,
                    "ir_filenames": self.ir_filenames[index],
                    "visible_filenames": self.visible_filenames[index]}