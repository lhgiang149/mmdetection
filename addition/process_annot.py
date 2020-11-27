import numpy as np
import cv2
import os
import multiprocessing

def get_size(source , path):
    if source[-1]!='/': source += '/'
    name = source + path
    image = cv2.imread(name)
    H,W,_ = image.shape
    return H, W, name

if __name__ == "__main__":
    while(True):
        in_path = input('Where do you store your image folder: ')
        assert os.path.isdir(in_path)
        image_path = os.listdir(in_path)
        try:
            get_size(in_path,image_path[0])
            break
        except:
            print('Wrong path!!')
    with open('annotation.txt', 'w') as f:
        for path in image_path:
            H,W, name = get_size(in_path, path)
            f.write('#\n')
            f.write(name + '\n')
            f.write('%d %d\n'%(W,H))
            f.write(('1\n'))
            f.write('0 0 0 0 0\n')
