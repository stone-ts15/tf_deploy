import sys
import ssl
import json
import time
import base64
from urllib.request import Request, urlopen


if len(sys.argv) != 2:
    print('[Usage] image_path')
    exit()

img_path = sys.argv[1]


# host = 'https://ocrapi-ugc.taobao.com'
# path = '/ocrservice/ugc'
host = 'https://tysbgpu.market.alicloudapi.com'
path = '/api/predict/ocr_general'
appcode = 'd781896dda6c4ff9af52bc495293617c'
bodys = {}
url = host + path


def img2base64(path):
    return base64.encodestring(open(path, "rb").read()).decode("utf8")


t0 = time.time()
post_data = img2base64(img_path)
# use 'img' or 'url'
bodys['image'] = post_data
# bodys['url'] = 'https://i.stack.imgur.com/t3qWG.png'
bodys = json.dumps(bodys).encode('ascii')
request = Request(url, bodys)
request.add_header('Authorization', 'APPCODE ' + appcode)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
response = urlopen(request, context=ctx)
content = response.read()
content = json.loads(content.decode("utf-8"))
print('call aliyun api consumes {} s'.format(time.time() - t0))
words = content['ret']
for word in words:
    print(word['word'])
