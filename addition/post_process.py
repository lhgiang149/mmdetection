import pickle
import numpy as np
import json
import os

def load_pickle(path, detectoRS = True):
    with open(path,'rb') as f:
        data = pickle.load(f)
    all_image = np.zeros((0,4), dtype = np.int32)
    for k,i in enumerate(data):
        img = i[0]
        if len(img) == 0:
            this_box = np.array([[0,0,10,10]])
        else:
            idx = np.argmax(img[:,4])
            this_bbox = img[idx].astype(np.int32)
            this_box = np.expand_dims(this_bbox[:4], axis = 0)
        all_image = np.concatenate((all_image, this_box), axis = 0)
    if detectoRS:
        modify = all_image.copy()
        modify = modify.astype(np.float32)
        scale_x = 7680/1333
        scale_y = 2160/800
        modify[..., 0]*=scale_x
        modify[..., 1]*=scale_y
        modify[..., 2]*=scale_x
        modify[..., 3]*=scale_y
        all_image = modify.astype(np.int32)
    need_to_dump = dict()
    length = all_image.shape[0]
    for i in range(0,length):
        need_to_dump[str(i)] = [int(j) for j in all_image[i]]
    name = path.split('.')[0]
    os.remove(path)
    with open(name + '.json', 'w') as f:
        json.dump(need_to_dump, f)

if __name__ == "__main__":
    path = 'result/detectoRS.pkl'
    load_pickle(path)
    path = 'result/yolo.pkl'
    load_pickle(path)