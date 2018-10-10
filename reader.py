from PIL import Image
import io
import numpy as np

def read_image_dims(image_data):
    with Image.open(io.BytesIO(image_data)) as im:
        data = np.array(im)
        assert(len(data.shape) == 3 and data.shape[2] in (1, 3))
        return data.shape[:2]
        
