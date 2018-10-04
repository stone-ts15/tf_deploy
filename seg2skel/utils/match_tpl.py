import numpy as np
import imutils
import cv2
import glob


def compute_iou(bbox1, bbox2):
    x1, y1, h1, w1 = bbox1
    x2, y2, h2, w2 = bbox2
    x_min = max(x1, x2)
    y_min = max(y1, y2)
    x_max = min(x1 + w1, x2 + w2)
    y_max = min(y1 + h1, y2 + h2)
 
    area1 = h1 * w1
    area2 = h2 * w2
    area12 = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    iou = area12 / float(area1 + area2 - area12)
    return iou


def create_template(tpl_path):
    tpl = cv2.imread(tpl_path)
    if tpl is None:
        raise ValueError('Cannot create tpl from {}'.format(tpl_path))
    tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    tpl = cv2.Canny(tpl, 50, 200)
    return tpl


def match_template(img, template, min_ratio=0.2, max_ratio=1, interp=20, auto=False, suffix='detect_arrow'):
    if isinstance(img, str):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tH, tW = template.shape[:2]
    H, W = gray.shape
    if auto:
        ratio = max(tH / H, tW / W)
        min_ratio = ratio
        max_ratio = ratio + 1
        interp = 10
    found = None
    for scale in np.linspace(min_ratio, max_ratio, interp)[::-1]:
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        r = gray.shape[1] / float(resized.shape[1])
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
 
    # cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
    # cv2.imshow("detected", img)
    # cv2.waitKey(0)
    # cv2.imwrite(img_path[:-4]+'_'+suffix+'_png', img)

    res = (startX, startY, endY - startY, endX - startX)
    return res


def match_multi_template(img, template, num=10, radius=10, min_ratio=0.2, max_ratio=5, interp=20, auto=False, suffix='detect_star'):
    if isinstance(img, str):
        img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tH, tW = template.shape[:2]
    H, W = gray.shape
    if auto:
        ratio = max(tH / H, tW / W)
        min_ratio = ratio
        max_ratio = ratio + 1
        interp = 10
    found = []
    for scale in np.linspace(min_ratio, max_ratio, interp)[::-1]:
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        r = gray.shape[1] / float(resized.shape[1])
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        for _ in range(num):
            _, maxval, _, maxloc = cv2.minMaxLoc(result)
            cv2.circle(result, maxloc, radius, (0.0), -1)
            found.append((maxval, maxloc, r))
    found = sorted(found, key=lambda x: -x[0])
    results = []
    # for f in found[:num]:
    #     _, maxLoc, r = f
    #     (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    #     (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    #     results += [(startX, startY, endY - startY, endX - startX)]
    for f in found:
        _, maxLoc, r = f
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        bbox = (startX, startY, endY - startY, endX - startX)
        nms = False
        for result in results:
            iou = compute_iou(bbox, result)
            if iou > 0.5:
                nms = True
        if not nms:
            results += [bbox]
            # cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            if len(results) >= num:
                break
    # cv2.imshow("detected", img)
    # cv2.waitKey(0)
    # cv2.imwrite(img_path[:-4]+'_'+suffix+'.png', img)
    return results


if __name__ == '__main__':
    import time
    photos = glob.glob('assets/lianjia/800p/*.png')
    # photos = ['1.png']
    if 0:
        template = create_template('assets/match_tpl/hstar.png')
        for fpath in photos:
            r = match_multi_template(fpath, template, num=2)
            print(r)
    if 0:
        template = create_template('assets/match_tpl/star.png')
        for fpath in photos:
            t0 = time.time()
            match_multi_template(fpath, template, suffix='detect_star')
            print('{} s'.format(time.time() - t0))
    template = create_template('assets/match_tpl/arrow.png')
    for fpath in photos:
        t0 = time.time()
        match_template(fpath, template)
        print('{} s'.format(time.time() - t0))
