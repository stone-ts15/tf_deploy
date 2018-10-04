# Apollo-v2
Algorithms for automatic home recognition.

## 环境
Python >= 3.3.x

```bash
pip isntall -r requirements.txt
```

Follow the [instruction](https://www.tensorflow.org/install/install_linux) to install `tensorflow`
To activate the environemnt
```bash
export PATH="/home/ec2-user/miniconda3/bin:$PATH"
export PYTHONPATH=$PYTHONPATH:/home/ec2-user/program/seg/:/home/ec2-user/program/seg/slim:/home/ec2-user/program/seg2skel
source activate tensorflow
```
or
```bash
source scripts/source.sh
```

## 链家户型图识别
```bash
cd $APOLLO_ROOT
python lianjia.py
```

### 使用Google OCR API
```bash
export GOOGLE_APPLICATION_CREDENTIALS=assets/google_api/home_plus.json
```
