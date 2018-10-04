import sys
import time
import text

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('[Usage] image_path')
        exit()

    img_path = sys.argv[1]
    try:
        t0 = time.time()
        ret = text.ocr_space_file(img_path)
        print(ret)
        t1 = time.time()
        print('ocr.space consumes {:.2f} s'.format(t1 - t0))
    except Exception as e:
        print('[Error] ocr.space:', e)

    try:
        t0 = time.time()
        ret = text.ocr_google_file(img_path)
        print(ret)
        t1 = time.time()
        print('ocr.google consumes {:.2f} s'.format(t1 - t0))
    except Exception as e:
        print('[Error] ocr.google:', e)

    try:
        t0 = time.time()
        ret = text.ocr_youtu_file(img_path)
        print(ret)
        t1 = time.time()
        print('ocr.youtu consumes {:.2f} s'.format(t1 - t0))
    except Exception as e:
        print('[Error] ocr.youtu:', e)
