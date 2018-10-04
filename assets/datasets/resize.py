import os
import cv2

fn = 'filelist.txt'
ifolder = 'data_hp'
ofolder = 'data_x2'

with open(fn) as f:
    for x in f.readlines():
        print(x.strip())
        ipath = os.path.join(ifolder, x.strip())
        opath = os.path.join(ofolder, x.strip())
        if not os.path.exists(os.path.dirname(opath)):
            os.makedirs(os.path.dirname(opath))
        img = cv2.imread(ipath)
        img = cv2.resize(img, None, fx=1.5, fy=1.5)
        cv2.imwrite(opath, img)
