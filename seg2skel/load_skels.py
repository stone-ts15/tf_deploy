# -*- coding: utf-8
#
import os
import json
import glob
import argparse
import numpy as np
from headspring.model import Point, Wall, Window, Door, House
from headspring.model import WallType, WindowType, DoorType, DoorOpenDirection


def loadjson(fpath):
    with open(fpath) as f:
        print('load plain skels from', fpath)
        info = json.load(f)
        return info


def dump2json(ofpath, data):

    with open(ofpath, 'w') as of:
        print('dump skels to', ofpath)
        json.dump(data, of)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate2Homeplus')
    parser.add_argument('--segment-folder', default='/home/ec2-user/program/assets/datasets/segmentation_results/', type=str)
    args = parser.parse_args()

    e = 1

    photos = glob.glob(os.path.join(args.segment_folder, '*_skels.json'))
    label2modeltype = {
        'wall': WallType.BEARING,
        'wall_bay': WallType.BAY,
        'window_common': WindowType.COMMON,
        'window_french': WindowType.FRENCH,
        'door_single': DoorType.SINGLE,
        'door_sliding': DoorType.SLIDING,
        'railing': WallType.RAILING
    }
    for fpath in photos:
        pos = fpath.rfind('_')
        text_fpath = fpath[:pos] + '_text.json'
        ofpath = fpath[:pos] + '_house.json'
        text = loadjson(text_fpath)
        scale = 10
        if 'scales' in text:
            scales = text['scales']
            if len(scales) > 1:
                scales = np.array(scales)
                text_scale = np.median(scales)
                if text_scale > 5:
                    scale = text_scale
        print(scale)

        infos = loadjson(fpath)
        x_min = int(infos['meta']['x_min'] * scale)
        y_min = int(infos['meta']['y_min'] * scale)
        x_max = int(infos['meta']['x_max'] * scale)
        y_max = int(infos['meta']['y_max'] * scale)
        walls = []
        doors = []
        windows = []
        for key, lst in infos.items():
            if key not in label2modeltype:
                continue
            kind = label2modeltype[key]
            for obj in lst:
                if key != 'door_single':
                    p1, p2, thick = obj
                else:
                    p1, p2, direction, thick = obj
                y1, x1 = p1
                y2, x2 = p2
                p1 = Point(int(x1 * scale), int(y1 * scale))
                p2 = Point(int(x2 * scale), int(y2 * scale))
                thick = int(thick * scale)
                if key != 'door_single':
                    if p1 > p2:
                        p1, p2 = p2, p1
                    # if thick < 10:
                    #     factor = 1.5
                    # else:
                    #     factor = 2
                    # if p1.x == p2.x or (p2.y - p1.y) / (p2.x - p1.x) > 1 or (p2.y - p1.y) / (p2.x - p1.x) < -1:
                    #     # because image is 2000x1500, around 1.33 makeup
                    #     half_thick = int((thick - scale) / factor / 1.5) + 1
                    #     half_thick = 0
                    #     p1.y = max(p1.y - half_thick, y_min)
                    #     p2.y = min(p2.y + half_thick, y_max)
                    # # elif p1.y == p2.y:
                    # elif -1 <= (p2.y - p1.y) / (p2.x - p1.x) <= 1:
                    #     half_thick = int((thick - scale) / factor) + 1
                    #     half_thick = 0
                    #     p1.x = max(p1.x - half_thick, x_min)
                    #     p2.x = min(p2.x + half_thick, x_max)
                    # else:
                    #     pass
                if -e < p1.x - p2.x < e and -e < p1.y - p2.y < e:
                    continue
                if key == 'wall' or key == 'wall_bay' or key == 'railing':
                    wall = Wall(p1, p2, thick, kind=kind)
                    # print(wall.json())
                    if not wall.is_point():
                        walls.append(wall)
                    else:
                        raise
                elif key == 'window_common' or key == 'window_french':
                    window = Window(p1, p2, kind=kind)
                    # print(window.json())
                    windows.append(window)
                elif key == 'door_sliding':
                    door = Door(p1, p2, kind=kind)
                    # print(door.json())
                    doors.append(door)
                elif key == 'door_single':
                    if direction == 'C':
                        open_direction = DoorOpenDirection.CLOCKWISE
                    elif direction == 'AC':
                        open_direction = DoorOpenDirection.ANTI_CLOCKWISE
                    else:
                        raise ValueError('open_direction({}) can only be "C" or "AC".'.format(direction))
                    door = Door(p1, p2, kind=kind, open_direction=open_direction)
                    # print(door.json())
                    doors.append(door)
                    if p1 > p2:
                        p1, p2 = p2, p1
                    wall = Wall(p1, p2, thick, kind=WallType.BEARING)
                    walls.append(wall)
                else:
                    raise KeyError('Unknown key: {}'.format(key))

        house = House(walls=walls, doors=doors, windows=windows)
        dump2json(ofpath, house.json())
