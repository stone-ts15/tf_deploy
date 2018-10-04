import os
import boto3


def download(bucket, key, folder='assets/datasets/data/JPEGImages/'):
    try:
        local_fn = folder + key.split('/')[-2] + '.jpg'
        if not os.path.isfile(local_fn):
            bucket.download_file(key, local_fn)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            print("Error({}): {}".format(key, w))


s3 = boto3.resource('s3')

# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)

bucket = s3.Bucket('seehome-house-raw')
suffix = '.jpg'
# print(len([0 for _ in bucket.objects.filter(Prefix='lianjia/layout')]))
# for obj in bucket.objects.all():
cnt = 0
ofn_list = 'assets/datasets/data/ImageSets/Segmentation/val.txt'
with open(ofn_list, 'w') as of:
    for obj in bucket.objects.filter(Prefix='lianjia/layout'):
        if obj.key[-4:] != suffix:
            continue
        print(obj)
        download(bucket, obj.key)
        of.write(obj.key.split('/')[-2] + '\n')
        cnt += 1
        if cnt >= 30:
            break
