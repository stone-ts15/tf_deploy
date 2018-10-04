import io
import os
import sys
import ssl
import json
import base64
import requests
from urllib.request import Request, urlopen

""" Currently, we use the free API provided by ocr.space and tencent.
    1. ocr.space
        - Request/months: 25,000
        - File Size Limit: 1 MB
        - Rate Limit: 500 calls/DAY
        - Some requests will be denied directly for unknown reason.
        More information can be found at https://ocr.space/ocrapi.

    2. youtu.tencent
        - Rate Limit: 1000 calls/DAY
        - The v1.0 api is really chaos which has multiple versions with
            different certification methods.
        - The v2.0 api lacks of mataintenance, thus we have to hack the
            interface to do general ocr.
        - The servie is very slow and unstable. And the detection box is shifted.
        More information can be found at https://github.com/tencentyun/image-python-sdk-v2.0/.

    3. vision.google
        - We use the latest version(v0.33.0) which is accurate and fast.
        - It provides character-level bounding box and confidence.
        - It can detect and verify the vertical word.
        - The detected bbox is not as tight as the bbox given by ocr.space.
        - Rate Limit: 1,000 UNITS/MONTH (free)
        - Since google cloud recommend to use `APPLICATION CREDENTIALS`
            instead of `API_KEY` for safety reason. We keep the service account
            file under `/assets/google_api/home_plus.json`. Please make sure
            this file will not be made public.
        More information can be found at https://cloud.google.com/vision/docs/python-client-migration

    Mutual verification will be applied based on the results from two apis.
"""

# TODO:
"""
1. wrap the api with class
    - Provide 2 funcs.
        - call api
        - parse response
    - Parsed response is a `list` where each element
        is a tuple with (number, left, top, height, width).
    - Init client once
"""


##################
#                #
#   ocr.aliyun   #
#                #
##################


ALIYUN_HOST = 'https://ocrapi-ugc.taobao.com'
ALIYUN_PATH = '/ocrservice/ugc'
ALIYUN_APPCODE = 'd781896dda6c4ff9af52bc495293617c'
ALIYUN_URL = ALIYUN_HOST + ALIYUN_PATH


def img2base64(path):
    return base64.encodestring(open(path, "rb").read()).decode("utf8")


def ocr_aliyun_file(img_path):
    bodys = {}
    post_data = img2base64(img_path)
    bodys['img'] = post_data
    bodys = json.dumps(bodys).encode('ascii')
    request = Request(ALIYUN_URL, bodys)
    request.add_header('Authorization', 'APPCODE ' + ALIYUN_APPCODE)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    response = urlopen(request, context=ctx)
    content = response.read()
    return content


##################
#                #
#   ocr.space    #
#                #
##################


API_KEY = '33de8d76c788957'


def ocr_space_file(filename, overlay=True, api_key=API_KEY, language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    try:
        with open(filename, 'rb') as f:
            r = requests.post('https://api.ocr.space/parse/image',
                              files={filename: f},
                              data=payload,
                              )
        return r.content.decode()
    except Exception as e:
        print('Calling ocr_space_file error:', e)
        return '{}'


def parse_ocr_space(results):

    def parse_word(w):
        n = w['WordText']
        x = w['Left']
        y = w['Top']
        h = w['Height']
        w = w['Width']
        return list(map(int, [n, x, y, h, w]))

    key = 'ParsedResults'
    parsed = []
    for r in results:
        r = json.loads(r)
        if key not in r or 'TextOverlay' not in r[key][0]:
            parsed.append([])
            continue
        r = r[key][0]['TextOverlay']
        p = []
        for l in r['Lines']:
            for w in l['Words']:
                try:
                    p.append(parse_word(w))
                except:
                    pass
        parsed.append(p)
    return parsed



##################
#                #
# youtu.tencent  #
#                #
##################


APP_ID = '1257137079'
SECRET_ID = 'AKIDc8S9i6PLCsMfnmTGl5ECyZ0KUTHtubJw'
SECRET_KEY = 'jHepohPvFi4FIBtQn0386va4MTODvSp3'
BUCKET = 'test'


def ocr_youtu_file(filename):
    from qcloud_image import Client, CIFiles
    from ._ext_qcloud_image import general_ocr
    appid = APP_ID
    secret_id = SECRET_ID
    secret_key = SECRET_KEY
    bucket = BUCKET
    client = Client(appid, secret_id, secret_key, bucket)
    client.use_http()
    client.set_timeout(30)
    # print(dir(client))
    return general_ocr(client, CIFiles([filename]))


def parse_ocr_youtu(results):

    def parse_word(w):
        n = w['itemstring']
        c = w['itemcoord']
        x = c['x']
        y = c['y']
        h = c['height']
        w = c['width']
        return [n] + list(map(int, [x, y, h, w]))

    parsed = []
    th = 0.9
    for r in results:
        r = r['data']
        p = []
        if len(r) > 0:
            for item in r['items']:
                try:
                    prob = 1
                    for w in item['words']:
                        prob *= w['confidence']
                    if prob < th:
                        continue
                    p.append(parse_word(item))
                except:
                    pass
        parsed.append(p)
    return parsed



##################
#                #
# vision.google  #
#                #
##################


GOOGLE_APPLICATION_CREDENTIALS = '/home/ec2-user/program/seg2skel/assets/google_api/home_plus.json'

def ocr_google_file(filename):
    cred = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if cred is None:
        print('[Error] To activate google API, please execute the following commands in bash:')
        print('export GOOGLE_APPLICATION_CREDENTIALS={}'.format(GOOGLE_APPLICATION_CREDENTIALS))
        return

    from google.cloud import vision
    from google.cloud.vision import types

    client = vision.ImageAnnotatorClient()

    with io.open(filename, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.document_text_detection(image=image)

    return response


def parse_ocr_google(results):

    def parse_word(w):

        # we use loop to find (left, top) adn (right, bottom),
        # because the order of vertices will change
        def get_left_top(vs):
            min_x, min_y = 1e6, 1e6
            for v in vs:
                min_x = min(v.x, min_x)
                min_y = min(v.y, min_y)
            return {'x': min_x, 'y': min_y}

        def get_right_bottom(vs):
            max_x, max_y = 0, 0
            for v in vs:
                max_x = max(v.x, max_x)
                max_y = max(v.y, max_y)
            return {'x': max_x, 'y': max_y}

        if len(w.symbols) > 4:
            # it may contain more than a word in one detected word
            text = ''.join([
                symbol.text for symbol in w.symbols
            ])
            texts = []
            bboxs = []
            text = ''
            start_i = 0
            for i, symbol in enumerate(w.symbols):
                if i == start_i:
                    # add top_left
                    bbox = [get_left_top(symbol.bounding_box.vertices)]
                text += symbol.text
                if not symbol.text.isdigit() \
                    or i == len(w.symbols) - 1:
                    if len(text) > 1:
                        # add bottom_lefet
                        texts += [text]
                        bbox.append(get_right_bottom(symbol.bounding_box.vertices))
                        bboxs.append(bbox)
                    start_i = i + 1
                    text = ''
                    bbox = []
            p = []
            for text, bbox in zip(texts, bboxs):
                try:
                    l, t = bbox[0]['x'], bbox[0]['y']
                    r, b = bbox[1]['x'], bbox[1]['y']
                    w = r - l
                    h = b - t
                    assert w > 0 and h > 0
                    p.append([int(text), l, t, h, w])
                except:
                    pass
            return p
        else:
            text = ''.join([
                symbol.text for symbol in w.symbols
            ])
            ## confidence can be used to filter out results
            # for symbol in w.symbols:
            #     print('\tSymbol: {} (confidence: {})'.format(
            #         symbol.text, symbol.confidence))
            left_top = get_left_top(w.bounding_box.vertices)
            right_bottom = get_right_bottom(w.bounding_box.vertices)
            l, t = left_top['x'], left_top['y']
            r, b = right_bottom['x'], right_bottom['y']
            w = r - l
            h = b - t
            assert w > 0 and h > 0
            if text == '米' or \
               text == '*' or \
               text == '不':
                return [[text, l, t, h, w]]
            else:
                return [[int(text), l, t, h, w]]

    parsed = []
    for r in results:
        if r is None:
            continue
        p = []
        for page in r.full_text_annotation.pages:
            for block in page.blocks:
                # print('\nBlock confidence: {}\n'.format(block.confidence))
                for paragraph in block.paragraphs:
                    # print('Paragraph confidence: {}'.format(
                    #     paragraph.confidence))
                    for w in paragraph.words:
                        try:
                            p.append(*parse_word(w))
                        except:
                            pass
        parsed.append(p)
    return parsed


def merge_ocr_result(results):
    """ Merge the result of parsed google results
        0. Filter out the number smaller than 100 or larger than 10000
        1. If the number is the same, we choose the one with smaller bounding box and mark it with high confidence.
        2. If the number is different, we choose the one with smaller number and mark it with low confidence.
    """

    def area(h, w):
        return h*w

    def overlap(bbox1, bbox2):
        l1, t1, h1, w1 = bbox1
        l2, t2, h2, w2 = bbox2
        r1 = l1 + w1
        r2 = l2 + w2
        b1 = t1 + h1
        b2 = t2 + h2
        dx = min(r1, r2) - max(l1, l2)
        dy = min(b1, b2) - max(t1, t2)
        if dx > 0 and dy > 0:
            return area(dx, dy)
        else:
            return 0

    l_th = 1e2
    h_th = 1e4
    o_th = 1e2
    refined = []
    bboxs = {}
    cnt = 0
    for r in results:
        n, l, t, h, w = r
        if isinstance(n, str):
            # uncomment following line to include the bbox of '*'
            # refined.append(r)
            continue
        if n < l_th or n > h_th:
            continue
        redundant = False
        for idx in bboxs:
            ovlp = overlap([l, t, h, w], bboxs[idx][1:])
            if  ovlp < o_th:
                continue
            else:
                redundant = True
                if n == bboxs[idx][0]:
                    confidence = 1 # place-holder
                    if area(h, w) < area(*bboxs[idx][3:]):
                        bboxs[idx] = [n, l, t, h, w]
                elif n < bboxs[idx][0]:
                        confidence = 0.5 # place-holder
                        bboxs[idx] = [n, l, t, h, w]
                else:
                    pass
        if not redundant:
            bboxs[cnt] = [n, l, t, h, w]
            cnt += 1
    assert cnt == len(bboxs)
    for idx in bboxs:
        refined.append(bboxs[idx])
    print('bbox has been refined from {} to {}.'\
            .format(len(results), len(bboxs)))

    return refined


def analyze_grid_by_bbox(results):

    def group_by_lines(lst, key_axis, sort_axis, th=10):
        assert key_axis == 1 or key_axis == 2
        assert sort_axis == 1 or sort_axis == 2
        lines = {}
        for r in lst:
            t = r[key_axis]
            is_new_line = True
            for key in lines:
                if abs(t - key) < th:
                    is_new_line = False
                    insert_by_ascend_order(lines[key], r, axis=sort_axis)
            if is_new_line:
                lines[t] = [r]
        return lines

    def insert_by_ascend_order(lst, r, axis):
        assert axis == 1 or axis == 2
        lst.append(r)
        x = r[axis]
        for i, l in enumerate(lst):
            if x >= l[axis]:
                continue
            for j in range(len(lst) - 2, i - 1, -1):
                lst[j + 1] = lst[j]
            lst[i] = r
            break
        return lst

    def analyze_grid(lines, axis):
        # 1 denotes horizon, 2 denotes vertical
        # -1 denotes width, -2 denotes height
        assert axis == 1 or axis == 2
        point = {}
        for key in lines:
            point[key] = []
            lst = []
            for l in lines[key]:
                if isinstance(l[0], int):
                    lst.append(l)
                    continue
                point[key].append(int(l[axis] + l[-axis]/2))

            for i in range(len(lst) - 1):
                # rb denotes right or bottom line
                rb = lst[i][axis] + lst[i][-axis]
                d = lst[i + 1][axis] - rb
                assert d > 0
                ratio = lst[i][0]*1./(lst[i][0] + lst[i+1][0])
                # add left/top point
                if i == 0:
                    point[key].append(int(lst[i][axis] - d*ratio))
                # add intermedian point
                point[key].append(int(rb + d*ratio))
                # add right/bottm point
                if i == len(lst) - 2:
                    point[key].append(int(lst[i+1][axis] + lst[i+1][-axis] + d*(1-ratio)))
        return point


    # group by direction
    horizon = []
    vertical = []
    for r in results:
        _, _, _, h, w = r
        if h < w:
            horizon += [r]
        else:
            vertical += [r]
    # group by lines
    hlines = group_by_lines(horizon, key_axis=2, sort_axis=1)
    vlines = group_by_lines(vertical, key_axis=1, sort_axis=2)
    # analyze grid
    hpoints = analyze_grid(hlines, axis=1)
    vpoints = analyze_grid(vlines, axis=2)
    return hpoints, vpoints


def analyze_panel_by_bbox(results, img_w, img_h):

    def group_by_lines(lst, key_axis, sort_axis, th=10):
        assert key_axis == 1 or key_axis == 2
        assert sort_axis == 1 or sort_axis == 2
        lines = {}
        for r in lst:
            t = r[key_axis]
            is_new_line = True
            for key in lines:
                if abs(t - key) < th:
                    is_new_line = False
                    insert_by_ascend_order(lines[key], r, axis=sort_axis)
            if is_new_line:
                lines[t] = [r]
        return lines

    def insert_by_ascend_order(lst, r, axis):
        assert axis == 1 or axis == 2
        lst.append(r)
        x = r[axis]
        for i, l in enumerate(lst):
            if x >= l[axis]:
                continue
            for j in range(len(lst) - 2, i - 1, -1):
                lst[j + 1] = lst[j]
            lst[i] = r
            break
        return lst

    def get_lines_min_max(lines, axis):
        min_ = 1e6
        max_ = -1e6
        for key in lines:
            for l in lines[key]:
                min_ = min(min_, l[axis])
                max_ = max(max_, l[axis])
        return min_, max_

    def analyze_panel(lines, axis, g_min, g_max):
        # 1 denotes horizon, 2 denotes vertical
        # -1 denotes width, -2 denotes height
        assert axis == 1 or axis == 2
        panels = []
        for key in lines:
            lst = lines[key]
            _, x_min, y_min, h, w = lst[0]
            x_max, y_max = x_min + w, y_min + h
            for bbox in lst[1:]:
                _, x, y, h, w = bbox
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            offset = 0
            if axis == 1:
                x_min = g_min
                x_max = g_max
            else:
                y_min = g_min
                y_max = g_max
            # panels.append([len(lst),
            panels.append([lst,
                           x_min - offset,
                           y_min - offset,
                           y_max - y_min + offset,
                           x_max - x_min + offset])

        return panels

    # group by direction
    horizon = []
    vertical = []
    for r in results:
        _, _, _, h, w = r
        if h < w:
            horizon += [r]
        else:
            vertical += [r]
    # group by lines
    hlines = group_by_lines(horizon, key_axis=2, sort_axis=1)
    vlines = group_by_lines(vertical, key_axis=1, sort_axis=2)
    # get min max
    if len(vlines) > 0:
        x_min, x_max = get_lines_min_max(vlines, axis=1)
    else:
        x_min, x_max = 0, img_w
    if len(hlines) > 0:
        y_min, y_max = get_lines_min_max(hlines, axis=2)
    else:
        y_min, y_max = 0, img_h
    # analyze panels
    hpanels = analyze_panel(hlines, axis=1, g_min=x_min, g_max=x_max)
    vpanels = analyze_panel(vlines, axis=2, g_min=y_min, g_max=y_max)
    return hpanels, vpanels
