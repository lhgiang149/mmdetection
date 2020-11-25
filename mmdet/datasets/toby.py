import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset
import cv2
from .pipelines import TobyRead


@DATASETS.register_module()
class TobyDataset(CustomDataset):
    CLASSES = ('ROI')

    def __init__(self, *args, **kwags):
        super(TobyDataset, self).__init__(*args, **kwags)
        self.pipeline = TobyRead(self.input_pipeline)

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for i, ann_line in enumerate(ann_list):
            if ann_line != '#':
                continue

            img_shape = ann_list[i + 2].split(' ')
            width = int(img_shape[0])
            height = int(img_shape[1])
            bbox_number = int(ann_list[i + 3])

            anns = ann_line.split(' ')
            bboxes = []
            labels = []
            for anns in ann_list[i + 4:i + 4 + bbox_number]:
                anns = anns.split(' ')
                bboxes.append([float(ann) for ann in anns[:4]])
                labels.append(int(anns[4]))

            data_infos.append(
                dict(
                    filename=ann_list[i + 1],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64))
                ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    # def __getitem__(self, idx):
    #     """Get training/test data after pipeline.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         dict: Training/test data (with annotation if `test_mode` is set \
    #             True).
    #     """
    #     add_path = '/home/../../data3/giangData/image_vol1_Sejin/%d.png'%(idx)
    #     if self.test_mode:
    #         data = self.prepare_test_img(idx)
    #         data['img'] = self.add_channel(data['img'].data, add_path, gray = True)
    #         return data
    #         # return self.prepare_test_img(idx)
    #     while True:
    #         data = self.prepare_train_img(idx)
    #         data['img'] = self.add_channel(data['img'].data, add_path, gray = True)
    #         if data is None:
    #             idx = self._rand_another(idx)
    #             continue
    #         return data

    # def add_channel(self, image, addition = '', gray = True):
    #     # bpo_path: image three channel with ball-people-others in order
    #     print('Type: ', type(image))
    #     try:
    #         print('shape: ', image.shape)
    #     except :
    #         print('shape: ', image.size)
    #     if not addition:
    #         return image
    #     if gray:
    #         add_image = cv2.imread(addition, 0)
    #     else:
    #         add_image = cv2.imread(addition)
    #         add_image = addition[:,:,::-1] # faster than cvtColor
    #     # print('Type: ', self.pipeline(add_image))
    #     out = np.concatenate((image,add_image), axis = 2)
    #     return out