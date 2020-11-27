import numpy as np
import cv2
import os
import multiprocessing
from tqdm import tqdm
from shutil import copyfile
def pre_process(path):
    name = path[0] + path[1]
    image = cv2.imread(name)
    size = (1333,800)
    image = cv2.resize(image, (size[0],size[1]), interpolation = cv2.INTER_LINEAR)
    p1 = (501, 168)
    p2 = (859, 168)
    p3 = (1333,633)
    p4 = (670, 672)
    p5 = (0, 613)
    points = np.array([p1,p2,p3,p4,p5])
    mask = np.zeros((size[1],size[0]))
    cv2.fillConvexPoly(mask, points, 255)
    mask=mask.astype(np.uint8)
    image[:,:,0][mask==0]=0
    image[:,:,1][mask==0]=0
    image[:,:,2][mask==0]=0
    cv2.imwrite('./data/' + path[1], image)
    
if __name__ == "__main__":
    in_path = input('Where do you store your image folder: ')
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    image_path = os.listdir(in_path)
    base = [in_path]*len(image_path)
    if image_path != os.listdir('./data/'):
        max_core = multiprocessing.cpu_count()
        with (multiprocessing.Pool(max_core)) as f:
            f.map(pre_process, zip(base, image_path))
    with open('yolo_mmdetection_full.txt', 'r') as f:
        lines = f.readlines()
    with open('yolo.txt', 'w') as f:
        count = 0
        count_lines = 0
        while (count_lines < len(lines)):
            word = lines[count_lines]
            if word == '#\n':
                f.write(word)
                f.write(in_path + image_path[count] + '\n')
                count+=1
                count_lines+=2
            f.write(word)
            count_lines+=1

    save_path = input('Where do you want to store your json file: ')
    with open('temp.txt', 'w') as f:
        f.write(save_path)