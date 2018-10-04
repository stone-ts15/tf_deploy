# -*- coding: utf-8
#
import cv2
import numpy as np
import imutils
import math
import os


def debug_show(img, window_name='debug'):
    cv2.imshow(window_name, img)
    cv2.waitKey()


def debug_save(img, out_name='debug.png'):
    cv2.imwrite(out_name, img)


def debug_draw_rect(img, left_top, right_bottom, color=(0, 255, 0)):
    cv2.rectangle(img, left_top, right_bottom, color)


def debug_draw_line(img, p1, p2, color=(0, 255, 0), lineThickness=1):
    cv2.line(img, p1, p2, color, lineThickness)


def debug_draw_circle(img, p, radius, color=(0, 255, 0), thickness=1):
    cv2.circle(img, p, radius, color, thickness=thickness)


def debug_draw_text(img, text, left_bottom, chinese=False, fontColor=(0, 255, 0)):
    assert isinstance(text, str)
    if not chinese:
        font      = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        lineType  = 2

        cv2.putText(img, text,
            left_bottom,
            font,
            fontScale,
            fontColor,
            lineType)
    else:
        from PIL import Image, ImageDraw, ImageFont
        FONT_PATH = "fonts/stliti.ttf"
        if not os.path.isfile(FONT_PATH):
            print("Use `bash tests/font_download.sh` to download chinese font.")
            return
        font = ImageFont.truetype(FONT_PATH, 20, encoding="utf-8")

        pilimg = Image.fromarray(img)
        draw = ImageDraw.Draw(pilimg)
        draw.text(left_bottom, text, fontColor, font=font)
        img[:] = np.array(pilimg)


def imread(fpath):
    img = cv2.imread(fpath)
    if img is None:
        raise FileNotFoundError('Cannot find {}'.format(fpath))
    return img


def imread_transparent(fpath):
    assert fpath[-4:] == '.png'

    img4c = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
    alpha = img4c[:,:,3]
    rgb = img4c[:,:,:3]
    bg = np.ones_like(rgb, dtype=np.uint8) * 255

    factor = alpha[:,:,np.newaxis].astype(np.float32) / 255.0
    factor = np.concatenate((factor,factor,factor), axis=2)

    img = rgb.astype(np.float32) * factor \
        + bg.astype(np.float32) * (1 - factor)

    return img.astype(np.uint8)


def rotate(img, degree):
    return imutils.rotate_bound(img, degree)


def rotate_point(origin, point, degree):
    ox, oy = origin
    angle = math.radians(degree)

    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def lianjia_dewatermark(png_img, jpg_img):
    delta = np.int16(png_img) - np.int16(jpg_img)
    delta_abs = np.abs(delta)
    delta_mask = np.uint8((delta_abs[:, :, 0] > 60) | (delta_abs[:, :, 1] > 60) | (delta_abs[:, :, 2] > 60)) * 255
    # debug_show(delta_mask)
    delta_mask = cv2.dilate(delta_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    delta[delta_mask == 0] = 0
    # debug_show(np.uint8(np.abs(delta)))
    img = png_img - delta * 1.45
    img[img < 0] = 0
    img[img > 255] = 255
    # debug_show(np.uint8(img))
    return np.uint8(img)
