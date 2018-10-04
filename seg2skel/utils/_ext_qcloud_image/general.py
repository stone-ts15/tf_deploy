import os
import sys
import urllib
from qcloud_image import CIUrl, CIFile, CIBuffer, CIUrls, CIFiles, CIBuffers
from .conf import Conf
from .http import Http
from .error import Error


def general_ocr(self, pictures):
    ''' general ocr
    
    :param pictures: pictures for general OCR
    :type  pictures: CIFiles or CIBuffers or CIUrls
    '''
    if not isinstance(pictures, CIUrls) and not isinstance(pictures, CIFiles) and not isinstance(pictures, CIBuffers):
        return Error.json(Error.Param, 'param pictures must be instance of CIUrls or CIFiles or CIBuffers')
    if len(pictures) == 0:
        return Error.json(Error.Param, 'param pictures is empty')

    requrl = self._conf.build_url('/ocr/general')
    headers = {
        'Host': self._conf.host(),
        'Authorization': self._auth.get_sign(self._bucket),
        'User-Agent': Conf.get_ua(self._auth._appid),
    }
    files = {
        'appid': self._auth._appid,
        'bucket': self._bucket
    }
    
    if isinstance(pictures, CIUrls):
        headers['Content-Type'] = 'application/json'

        files['url_list'] = pictures
        return Http.post(requrl, headers=headers, data=json.dumps(files), timeout=self._conf.timeout())

    else:
        if isinstance(pictures, CIFiles):
            i=0
            for image in pictures:
                if sys.version_info < (3, 0) :
                    filename = urllib.quote(image)    
                    image = image.decode('utf-8')                             
                else:
                    filename = urllib.parse.quote(image)
                
                local_path = os.path.abspath(image)
                if not os.path.exists(local_path):
                    return Error.json(Error.FilePath, 'file '+image+' not exists')
                
                if not os.path.isfile(local_path):
                    return Error.json(Error.FilePath, local_path+' is not file')
                files['images['+str(i)+']'] = (filename, open(image,'rb'))
                files['image'] = (filename, open(image,'rb'))
                i = i+1

        else:
            i=0
            for buffer in pictures:
                files['images['+str(i)+']'] = buffer
                i = i+1

    return Http.post(requrl, headers=headers, files=files, timeout=self._conf.timeout())
