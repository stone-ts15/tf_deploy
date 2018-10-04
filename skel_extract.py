# -*- coding: utf-8
#
import os
import cv2
import glob
import json
import time
import argparse
import numpy as np
from collections import deque
# from seg2skel.utils import image as image_utils
# from seg2skel.utils import text as text_utils
# from seg2skel.utils import match_tpl as mtpl_utils
# from seg2skel.utils import skel as skel_utils

from utils import image as image_utils
from utils import text as text_utils
from utils import match_tpl as mtpl_utils
from utils import skel as skel_utils
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

class Skeleton(object):

    DEBUG = True
    OUTPUT_MID = False

    def __init__(self, debug=True):
        # labels are given by bgr
        self.labels = {
            'wall':             [0, 0, 255],
            'wall_bay':         [0, 255, 12],
            'window_common':    [255, 0, 0],
            'window_french':    [255, 150, 150],
            'door_single':      [255, 255, 0],
            'door_sliding':     [0, 255, 255],
            'railing':          [255, 0, 255],
        }
        self.DEBUG = debug
        if self.DEBUG:
            print('Skeleton DEBUG mode is ON')

    def extract(self, ori_img, seg_img, ofpath, scale=1):
        skels = {}
        for key, label in self.labels.items():
            masked = np.all(seg_img == label, axis=2) * 255
            if key == 'wall':
                for wall_key, wall_label in self.labels.items():
                    if wall_key == 'door_single' or wall_key.find('wall') >= 0:
                        continue
                    masked += np.all(seg_img == wall_label, axis=2) * 255
                ys, xs = masked.nonzero()
                skels['meta'] = {}
                skels['meta']['x_min'] = xs.min()
                skels['meta']['y_min'] = ys.min()
                skels['meta']['x_max'] = xs.max()
                skels['meta']['y_max'] = ys.max()
            masked = masked.astype(np.uint8)
            if key == 'door_single':
                # with Timer('find door_single'):
                edges, bboxs = self.find_single_door(masked)
            elif key == 'wall_bay':
                # with Timer('find wall_bay'):
                edges = self.find_wall_bay(masked)
            else:
                # with Timer('skeletonize {}'.format(key)):
                skel = skel_utils.skeletonize(masked)
                # with Timer('binary2edges'):
                edges, se_points, se_corners = self.binary2edges(skel)
                if key == 'wall' and self.DEBUG and self.OUTPUT_MID:
                    print('output mid-level results')
                    mid_res_img = ori_img.copy()
                    for edge in edges:
                        p1, p2 = edge
                        p1, p2 = self.edge2endpoints([p1, p2])
                        image_utils.debug_draw_line(mid_res_img, p1, p2)
                        for p in se_points:
                            p = (p[1], p[0])
                            image_utils.debug_draw_circle(mid_res_img, p, 2)
                        for p in se_corners:
                            p = (p[1], p[0])
                            image_utils.debug_draw_circle(mid_res_img, p, 2, color=(255, 0, 0))
                    cv2.imwrite(ofpath[:-5] + '_mid0.png', skel)
                    cv2.imwrite(ofpath[:-5] + '_mid.png', mid_res_img)
                print('refine edges')
                edges = self.refine_edges(edges)
                if key == 'wall':
                    print('more proposals')
                    edges = self.propose_edges(edges, masked)
                if key.find('wall') >= 0:
                    print('compute thickness')
                    min_thickness = 3
                    edges = self.compute_thickness(edges, masked, min_thickness)
                else:
                    edges = self.dummy_thickness(edges)
                if key == 'railing':
                    walls = np.all(seg_img == self.labels['wall'], axis=2) * 255
                    walls = walls.astype(np.uint8)
                    edges = self.inside_constrain(edges, walls)
            # skels[key] = self.rescale(edges, scale)
            skels[key] = edges
        skels['door_single'], skels['wall'] = self.global_rectify(skels['door_single'], skels['wall'])
        skels['wall'] = self.refine_connection(skels['wall'])
        self.dump2json(ofpath, skels)
        if self.DEBUG:
            for key, edges in skels.items():
                if key == 'meta':
                    continue
                if key == 'door_single':
                    for edge in edges:
                        p1, p2, direction, _ = edge
                        p1, p2 = self.edge2endpoints([p1, p2])
                        image_utils.debug_draw_line(ori_img, p1, p2)
                        image_utils.debug_draw_circle(ori_img, p1, 2, color=(0, 0, 255))
                        image_utils.debug_draw_text(ori_img, direction, p1)
                    # for bbox in bboxs:
                    #     x, y, w, h = bbox
                    #     p1 = (x, y)
                    #     p2 = (x+w, y+h)
                    #     image_utils.debug_draw_rect(ori_img, p1, p2)
                elif key == 'wall':
                    for edge in edges:
                        p1, p2, thick = edge
                        p1, p2 = self.edge2endpoints([p1, p2])
                        image_utils.debug_draw_line(ori_img, p1, p2)
                        x1, y1 = p1
                        x2, y2 = p2
                        mid_x = int((x1 + x2) / 2)
                        mid_y = int((y1 + y2) / 2)
                        half_thick = (thick - 1) / 2
                        if half_thick == 0:
                            continue
                        if x1 == x2:
                            p1 = (int(mid_x - half_thick), mid_y)
                            p2 = (int(mid_x + half_thick), mid_y)
                        elif y1 == y2:
                            p1 = (mid_x, int(mid_y - half_thick))
                            p2 = (mid_x, int(mid_y + half_thick))
                        else:
                            k = (y2 - y1) / (x2 - x1)
                            kk = - 1./k
                            s = np.sqrt(thick**2 / (kk**2 + 1))
                            tx1 = int(mid_x + s / 2)
                            tx2 = int(mid_x - s / 2)
                            ty1 = int(mid_y + s*kk / 2)
                            ty2 = int(mid_y - s*kk / 2)
                            p1 = (tx1, ty1)
                            p2 = (tx2, ty2)
                        image_utils.debug_draw_line(ori_img, p1, p2, (0, 0, 255))
                        # length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        # image_utils.debug_draw_text(ori_img, str(length), p1)
            cv2.imwrite(ofpath[:-5] + '.png', ori_img)
        return skels

    @staticmethod
    def check_ners(idx, binary):
        ners = []
        cy, cx = idx
        for y in range(cy - 1, cy + 2):
            for x in range(cx - 1, cx + 2):
                if y == cy and x == cx:
                    continue
                if binary[y][x] > 0:
                    ners.append([y, x])
        types = set()
        for ner in ners:
            ny, nx = ner
            if nx == cx:
                k = 'v'
            else:
                k = (ny - cy) // (nx - cx)
                k = str(k)
            types.add(k)
        type_ = 'unknown'
        if len(types) == 1:
            type_ = types.pop()
        elif len(types) >= 2:
             type_ = 'corner'
        return (len(ners), ners, type_)

    @staticmethod
    def idx2key(idx):
        return '-'.join([str(i) for i in idx])

    @classmethod
    def find_recursive(cls, start, ners, se_set, idx2ners, p2visited, points_):
        # TODO: To be deprecated
        for idx in ners:
            key = cls.idx2key(idx)
            if key != start and (key in se_set) and (idx not in points_):
                # end-points or corner can be visited more than one time
                p2visited[key] += 1
                points_.append(idx)
            else:
                if p2visited[key] > 0:
                    continue
                p2visited[key] = 1
                cls.find(start, idx2ners[key][1], se_set, idx2ners, p2visited, points_)

    @classmethod
    def find(cls, start, ners, se_set, idx2ners, p2visited, points_):
        stack = []
        stack.extend(ners)
        while len(stack) > 0:
            idx= stack[len(stack) - 1]
            stack.remove(idx)
            key = cls.idx2key(idx)
            if key != start and (key in se_set) and (idx not in points_):
                # end-points or corner can be visited more than one time
                p2visited[key] += 1
                points_.append(idx)
            else:
                if p2visited[key] > 0:
                    continue
                p2visited[key] = 1
                stack.extend(idx2ners[key][1])

    @classmethod
    def binary2edges(cls, binary):
        edges = []
        idxs = np.argwhere(binary)
        idx2ners = {}
        se_points = []
        se_corners = []
        se_set = set()
        p2visited = {}
        for i, idx in enumerate(idxs):
            idx = list(idx)
            key = cls.idx2key(idx)
            size, ners, type_ = cls.check_ners(idx, binary)
            if size == 1:
                se_points.append(idx)
                se_set.add(key)
            if type_ == 'corner':
                se_corners.append(idx)
                se_set.add(key)
            idx2ners[key] = (size, ners, type_)
            p2visited[key] = 0
        # dfs, terminated when encounter end-point or corner
        edges = []
        for points in [se_points, se_corners]:
            for idx in points:
                start_key = cls.idx2key(idx)
                p2visited[start_key] += 1
                size, ners, type_ = idx2ners[start_key]
                assert size == 1 or type_ == 'corner'
                points_ = []
                cls.find(start_key, ners, se_set, idx2ners, p2visited, points_)
                for point in points_:
                    edges.append([idx, point])
        if cls.DEBUG:
            cnt = 0
            for _, v in p2visited.items():
                if v == 0:
                    cnt += 1
            print('#points: {}, #edges: {}, #not visited points: {}'.\
                    format(len(idxs), len(edges), cnt))
        return edges, se_points, se_corners

    @staticmethod
    def edge2endpoints(edge):
        p1, p2 = edge
        assert len(p1) == 2 and len(p2) == 2
        return (p1[1], p1[0]), (p2[1], p2[0])

    @staticmethod
    def compute_k(edge):
        p1, p2 = edge
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            k = [1, 0, -x1]
            if y1 > y2:
                edge = [(x1, y2), (x2, y1)]
        else:
            if y1 == y2:
                # we add this special case to avoid -0.0 and 0.0
                k = 0
            else:
                k = (y2 - y1) / (x2 - x1)
            b = (0 - x1) * k + y1
            k = [-k, 1, -b]
            if x1 > x2:
                edge = [(x2, y2), (x1, y1)]
        return k, edge

    @staticmethod
    def insert_by_descend_length(lst, d, edge, k):
        lst.append([d, edge, k])
        for i, l in enumerate(lst):
            if l[0] >= d:
                continue
            for j in range(len(lst) - 2, i - 1, -1):
                lst[j + 1] = lst[j]
            lst[i] = [d, edge, k]
            break
        return lst

    @staticmethod
    def dist_points(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx**2 + dy**2)

    @classmethod
    def refine_edges(cls, edges):

        def dist_lines(k1, k2):
            a1, b1, c1 = k1
            a2, b2, c2 = k2
            if a1 == a2 and b1 == b2:
                return abs(c1 - c2) / np.sqrt(a1**2 + b1**2)
            else:
                return -1

        def k2key(k):
            return '{}-{}'.format(k[0], k[1])

        def isclose(e1, e2, th):
            p11, p12 = e1
            p21, p22 = e2
            if cls.dist_points(p11, p21) < th or \
               cls.dist_points(p11, p22) < th or \
               cls.dist_points(p12, p21) < th or \
               cls.dist_points(p12, p22) < th:
                return True
            else:
                return False

        def merge_two_lines(e1, e2, k):
            if k[:2] == [1, 0]:
                y_min = min(e1[0][1], e2[0][1])
                y_max = max(e1[1][1], e2[1][1])
                p11 = (e1[0][0], y_min)
                p12 = (e1[1][0], y_max)
            elif k[:2] == [0, 1]:
                x_min = min(e1[0][0], e2[0][0])
                x_max = max(e1[1][0], e2[1][0])
                p11 = (x_min, e1[0][1])
                p12 = (x_max, e1[1][1])
            else:
                a, b, c = k
                assert b != 0, "k: {}".format(k)
                x_min = min(e1[0][0], e2[0][0])
                x_max = max(e1[1][0], e2[1][0])
                y_min = - (a / b) * x_min - (c / b)
                y_max = - (a / b) * x_max - (c / b)
                p11 = (x_min, int(y_min))
                p12 = (x_max, int(y_max))
            return p11, p12

        def merge_short_edges(edges):
            size = len(edges)
            visited = set()
            idx = 0
            merged_edges = []
            # TODO: use precomputed index to optimize the bfs search
            while idx < size:
                if idx in visited:
                    idx += 1
                    continue
                queue = deque([idx])
                waiting_set = set()
                min_x, max_x = 1e6, 0
                min_ep, max_ep = None, None
                while queue:
                    idx1 = queue.popleft()
                    visited.add(idx1)
                    _, e1, _ = edges[idx1]
                    p1, p2= e1
                    x1, x2 = p1[0], p2[0]
                    if x1 < min_x:
                        min_x = x1
                        min_ep = p1
                    if x2 > max_x:
                        max_x = x2
                        max_ep = p2
                    merged = False
                    for th_ep_dist in [3, 4]:
                        for idx2 in range(idx1 + 1, size):
                            if idx2 in visited or idx2 in waiting_set:
                                continue
                            _, e2, _ = edges[idx2]
                            if isclose(e1, e2, th_ep_dist):
                                merged = True
                                waiting_set.add(idx2)
                                queue.append(idx2)
                        if merged:
                            break
                d = cls.dist_points(min_ep, max_ep)
                k, edge = cls.compute_k([min_ep, max_ep])
                merged_edges.append([d, edge, k])
                idx += 1
            return merged_edges

        th_line_len = 5 # too small will affect oblique line
        th_endpoint_dist = 5
        rf_edges = []
        short_edges = []
        group = {}
        for edge in edges:
            if len(edge) == 2:
                p1, p2 = edge
                d = cls.dist_points(p1, p2)
                k, edge = cls.compute_k(edge)
                p1, p2 = edge
            else:
                d, edge, k = edge
                p1, p2 = edge
            key = k2key(k)
            if d < th_line_len and k[:2] != [0, 1] and k[:2] != [1, 0]:
                # print('short line', p1, p2, d, k)
                short_edges.append([d, edge, k])
            else:
                if key not in group:
                    group[key] = []
                group[key] = cls.insert_by_descend_length(group[key], d, edge, k)
        for key, lines in group.items():
            rf_lines = []
            idx1 = 0
            size = len(lines)
            flags = np.zeros(size)
            while idx1 < size:
                if flags[idx1]:
                    idx1 += 1
                    continue
                d1, e1, k1 = lines[idx1]
                if d1 < th_line_len:
                    short_edges.append(lines[idx1])
                    flags[idx1] = 1
                    idx1 += 1
                    continue
                merged = False
                idx2 = size - 1
                while idx1 < idx2:
                    if flags[idx2]:
                        idx2 -= 1
                        continue
                    d2, e2, k2 = lines[idx2]
                    assert d1 >= d2, "{} vs {}".format(d1, d2)
                    d = dist_lines(k1, k2)
                    th_line_dist = 0.1 * d1
                    th_line_dist = min(th_line_dist, 5)
                    th_line_dist = max(th_line_dist, 1)
                    th_endpoint_dist = max(0.1 * d1, 5)
                    if d >= 0 and d <= th_line_dist and isclose(e1, e2, th_endpoint_dist):
                        merged = True
                        flags[idx2] = 1
                        p11, p12 = merge_two_lines(e1, e2, k1)
                        # update d1, e1
                        d1 = cls.dist_points(p11, p12)
                        e1 = [p11, p12]
                        lines[idx1] = [d1, e1, k1]
                        break
                    idx2 -= 1
                if not merged:
                    rf_lines.append(lines[idx1])
                    flags[idx1] = 1
                    idx1 += 1
            for i in range(size):
                if not flags[i]:
                    rf_lines.append(lines[i])
            rf_edges.extend(rf_lines)
        rf_short_edges = merge_short_edges(short_edges)
        rf_edges.extend(rf_short_edges)
        print('reduce short edges from {} to {}'.format(len(short_edges), len(rf_short_edges)))
        print('reduce edges from {} to {}'.format(len(edges), len(rf_edges)))
        return rf_edges

    @staticmethod
    def point_lt(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x1 < x2:
            return True
        elif x1 > x2:
            return False
        if y1 < y2:
            return True
        elif y1 > y2:
            return False
        return False

    @staticmethod
    def line_inside_masked(k, edge, masked):
        p1, p2 = edge
        x1, y1 = p1
        x2, y2 = p2
        a, b, c = k
        # check all points along the line (except endpoints)
        if b == 0:
            assert y1 <= y2 and x1 == x2
            for y in range(y1 + 1, y2):
                if not masked[x1, y]:
                    return False
        elif a == 0:
            assert x1 <= x2 and y1 == y2
            for x in range(x1 + 1, x2):
                if not masked[x, y1]:
                    return False
        else:
            assert x1 <= x2 or y1 <= y2
            for x in range(min(x1, x2), max(x1, x2)):
                y = int(- (1. * a / b) * x - 1. * c / b)
                if not masked[x, y]:
                    return False
            for y in range(min(y1, y2), max(y1, y2)):
                x = int(- (1. * b / a) * y - 1. * c / a)
                if not masked[x, y]:
                    return False
        return True


    @classmethod
    def propose_edges(cls, edges, masked):

        min_line_len = 5
        max_line_len = 150
        min_angle = np.tan(1 * np.pi / 180)
        max_angle = np.tan(89 * np.pi / 180)

        def propose(e1, e2, masked):
            p11, p12 = e1
            p21, p22 = e2
            edge_lst = [(p11, p21), (p11, p22), (p12, p21), (p12, p22)]
            rets = []
            for edge in edge_lst:
                p1, p2 = edge
                if not cls.point_lt(p1, p2):
                    continue
                k, _ = cls.compute_k(edge)
                if cls.line_inside_masked(k, edge, masked):
                    d = cls.dist_points(p1, p2)
                    # only propose long oblique line
                    if d > max_line_len and k[0] != 0 and k[1] != 0 and \
                        min_angle <= abs(k[1] / k[0]) <= max_angle:
                        rets.append([d, edge, k])
            return rets

        more_edges = []
        more_edges.extend(edges)
        tot = len(edges)
        for i in range(tot):
            d1, e1, k = edges[i]
            if (d1 <= min_line_len) or \
               (d1 >= max_line_len and (k[0] == 0 or k[1] == 0)):
                continue
            for j in range(i + 1, tot):
                d2, e2, _ = edges[j]
                if d2 <= min_line_len:
                    continue
                more_edges.extend(propose(e1, e2, masked))
        print('edges from {} to {}'.format(len(edges), len(more_edges)))
        return more_edges

    @staticmethod
    def point2edge(p, edge):
        p1, p2 = edge
        v1 = np.array([p[0] - p1[0], p[1] - p1[1]])
        v1_norm = np.linalg.norm(v1)
        v2 = np.array([p[0] - p2[0], p[1] - p2[1]])
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1 or v2_norm < 1:
            return 0
        v12 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        if np.dot(v1, v12) >= 0 and np.dot(v2, -v12) >= 0:
            v12_norm = np.linalg.norm(v12)
            cos = np.dot(v1, v12) / v1_norm / v12_norm
            if cos >= 1:
                return 0
            else:
                return v1_norm * np.sqrt(1 - cos**2)
        else:
            return min(v1_norm, v2_norm)

    @classmethod
    def point_inside_edge(cls, edge, p):
        p1, p2, thick = edge
        if thick == 0:
            return False
        x1, y1 = p1
        x2, y2 = p2
        xp, yp = p
        k, _ = cls.compute_k((p1, p2))
        a, b, c = k
        offset = 3
        half_thick = int((thick - 1) / 2) + offset
        if a == 0 or b == 0:
            x1 -= half_thick
            y1 -= half_thick
            x2 += half_thick
            y2 += half_thick
            if x1 <= xp <= x2 and y1 <= yp <= y2:
                return True
        else:
            # require the distance of point to line segment smaller than half_thick
            if cls.point2edge(p, (p1, p2)) < half_thick:
                return True
        return False

    @staticmethod
    def dummy_thickness(edges):
        th_len = 3
        thick_edges = []
        for d, (p1, p2), _ in edges:
            if d <= th_len:
                continue
            thick_edges.append([p1, p2, 0])
        return thick_edges

    @classmethod
    def compute_thickness(cls, edges, img, min_thickness):

        def is_contain(thick_edges, edge):
            p1, p2 = edge
            for e in thick_edges:
                if cls.point_inside_edge(e, p1) and\
                   cls.point_inside_edge(e, p2):
                    return True
            return False

        def get_thick(thicks, d):
            if len(thicks) == 0:
                return 0
            thick = np.ceil(np.median(thicks))
            th = int(0.1 * d)
            if thick > d + th:
                rf_thicks = []
                for thick in thicks:
                    if abs(d - thick) < th:
                        rf_thicks.append(thick)
                if len(rf_thicks) > 0:
                    thick = np.median(rf_thicks)
                else:
                    thick = (d + 1) / 2
            return 2*thick - 1

        th_len = 3
        thick_edges = []
        edges = sorted(edges, key=lambda x:-x[0])
        for i, edge in enumerate(edges):
            d, (p1, p2), k = edge
            if d <= th_len:
                continue
            if is_contain(thick_edges, edge=(p1, p2)):
                continue
            a, b, _ = k
            x1, y1 = p1
            x2, y2 = p2
            thicks = []
            # if (a == 1 and b == 0) or (-1 <= -a / b <= 1):
            if (a == 1 and b == 0) or -a / b < -1 or -a / b > 1:
                # assert x1 == x2, "p1: {} vs p2: {}, k: {}".format(p1, p2, k)
                my0 = int((y1 + y2) / 2)
                my1 = int((3*y1 + y2) / 4)
                my2 = int((y1 + 3*y2) / 4)
                for x, y, step in [(x1, y1, -1), (x2, y2, 1)]:
                    cnt = 0
                    while img[x, y] > 0:
                        cnt += 1
                        y += step
                    if cnt > 0:
                        thicks.append(cnt)
                for step in [-1, 1]:
                    for y in [my0, my1, my2, y1, y2]:
                        cnt = 0
                        x = x1
                        while x < img.shape[0] and y < img.shape[1] and img[x, y] > 0:
                            cnt += 1
                            x += step
                        if cnt > 0:
                            thicks.append(cnt)
                thick = get_thick(thicks, d)
            # elif a == 0 or -a / b < -1 or -a / b > 1:
            elif -1 <= -a / b <= 1:
                # assert y1 == y2, "p1: {} vs p2: {}, k : {}".format(p1, p2, k)
                mx0 = int((x1 + x2) / 2)
                mx1 = int((3*x1 + x2) / 4)
                mx2 = int((x1 + 3*x2) / 4)
                for x, y, step in [(x1, y1, -1), (x2, y2, 1)]:
                    cnt = 0
                    while img[x, y] > 0:
                        cnt += 1
                        x += step
                    if cnt > 0:
                        thicks.append(cnt)
                for step in [-1, 1]:
                    for x in [mx0, mx1, mx2, x1, x2]:
                        cnt = 0
                        y = y1
                        while x < img.shape[0] and y < img.shape[1] and img[x, y] > 0:
                            cnt += 1
                            y += step
                        if cnt > 0:
                            thicks.append(cnt)
                thick = get_thick(thicks, d)
            else:
                mx0 = int((x1 + x2) / 2)
                mx1 = int((3*x1 + x2) / 4)
                mx2 = int((x1 + 3*x2) / 4)
                my0 = int((y1 + y2) / 2)
                my1 = int((3*y1 + y2) / 4)
                my2 = int((y1 + 3*y2) / 4)
                for step in [-1, 1]:
                    for x0, y0 in [(mx0, my0), (mx1, my1), (mx2, my2), (x1, y1), (x2, y2)]:
                        cnt = 0
                        x, y = x0, y0
                        # print(x, y, img.shape, img[x, y])
                        while x < img.shape[0] and y < img.shape[1] and img[x, y] > 0:
                            cnt += 1
                            x += step
                            y = int((b / a) * (x - x0) + y0)
                            # print(b, a, y, y0, x, x0)
                        if cnt > 0:
                            thicks.append(cnt)
                thick = get_thick(thicks, d)
            if thick >= min_thickness:
                thick_edges.append([p1, p2, thick])
        # global optimize: remove unreasonable thick
        if len(thick_edges) > 1:
            scale = 1.5
            thickness = [thick for _, _, thick in thick_edges]
            thickness = int(np.median(thickness))
            for i, (p1, p2, thick) in enumerate(thick_edges):
                if thick > scale * thickness:
                    thick_edges[i] = [p1, p2, thickness]
        print('from {} original edges to compote {} thick edges.'.format(len(edges), len(thick_edges)))
        return thick_edges

    @classmethod
    def refine_connection(cls, edges):
        # edges are ranked by length
        # use long edge to refine connection of short edges

        def get_min(p, plst, j, flags, p1_min, p1_min_idx):
            dists = []
            for jk in range(len(plst)):
                if flags[j][jk]:
                    dists.append(1e6)
                else:
                    dists.append(cls.dist_points(p, plst[jk]))
            assert len(dists) == 2
            idx = np.argmin(dists)
            if dists[idx] < p_min:
                return dists[idx], (j, idx)
            return p1_min, p1_min_idx

        tot = len(edges)
        flags = np.zeros([tot, 2])
        for i in range(tot):
            p1, p2, thick = edges[i]
            th = thick
            for ik, p in enumerate([p1, p2]):
                flags[i, ik] = 1
                p_min = 1e6
                p_min_idx = -1
                for j in range(i + 1, tot):
                    p_min, p_min_idx = get_min(p, edges[j][:2], j, flags, p_min, p_min_idx)
                if 0 < p_min < th and p_min_idx != -1:
                    print('find!', p_min_idx, p_min, thick)
                    j, jk = p_min_idx
                    if not flags[j][jk]:
                        flags[j][jk] = 1
                        edges[j][jk] = p
        edges = edges[::-1]
        for i, wall in enumerate(edges):
            p1, p2, thick = wall
            for j, p in enumerate([p1, p2]):
                d, close_pt = cls.min_dist_to_walls(p, edges, ignore=[i])
                if 0 < d < 5*thick and close_pt is not None:
                    edges[i][j] = close_pt
                    # edges.append([edges[i][j], close_pt, thick])
        return edges

    @staticmethod
    def find_wall_bay(masked):
        _, contours, _ = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        edges = []
        for c in contours:
            y, x, h, w = cv2.boundingRect(c)
            if w < 10 and h < 10:
                continue
            if w > h:
                thick = h
                p1 = (x + int(thick / 2), int(y + h/2))
                p2 = (x + w - int(thick / 2), int(y + h/2))
            else:
                thick = w
                p1 = (int(x + w/2), y + int(thick / 2))
                p2 = (int(x + w/2), y + h - int(thick / 2))
            edges.append([p1, p2, thick])
        return edges

    @staticmethod
    def find_single_door(masked):

        def find_shaft(patch, p0, p1, p2, p3):
            shaft2edge = {
                0: [p0, p1, 'C'],
                1: [p1, p0, 'AC'],
                2: [p2, p3, 'C'],
                3: [p3, p2, 'AC'],
            }
            cnt = np.zeros(4)
            row, col = patch.shape
            for x in range(row):
                y1 = -1. * col / row * x + col
                y2 = col / row * x
                for y in range(col):
                    if patch[x][y] == 0:
                        continue
                    if y < y1:
                        cnt[0] += 1
                    elif y > y1:
                        cnt[2] += 1
                    else:
                        pass
                    if y < y2:
                        cnt[3] += 1
                    elif y > y2:
                        cnt[1] += 1
                    else:
                        pass
            shaft_idx = cnt.argmax()
            edge = shaft2edge[shaft_idx]
            return edge

        edges = []
        _, contours, _ = cv2.findContours(masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        bboxs = []
        uncertain_lst = []
        door_len_lst = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            door_len_lst.append(w)
            door_len_lst.append(h)
        door_len = int(np.median(door_len_lst))
        th_door_len = int(0.3 * door_len)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < th_door_len and h < th_door_len:
                # ignore small doors
                # print('small_door', w, h, door_len, th_door_len)
                continue
            if (h - door_len) > th_door_len or \
               (w - door_len) > th_door_len:
                uncertain_lst.append([x, y, w, h])
                continue
            p0 = (y, x)
            p1 = (y, x + w)
            p2 = (y + h, x + w)
            p3 = (y + h, x)
            patch = masked[y:y+h, x:x+w]
            edge = find_shaft(patch, p0, p1, p2, p3)
            edges.append(edge)
            bboxs.append([x, y, w, h])

        for r in uncertain_lst:
            x, y, w, h = r
            num = max(int(w / door_len), int(h / door_len))
            # TODO: search the triangle structure
            if num < 1 or num > 3:
                print('[warn] there are {} doors in the contour. (only support 2 doors curretly)'.format(num))
                continue
            area = np.zeros(4)
            idx2p = [
                (y, x),
                (y, x + w - door_len),
                (y + h - door_len, x + w - door_len),
                (y + h - door_len, x),
            ]
            for idx, p in enumerate(idx2p):
                y1, x1 = p
                y2, x2 = y1 + door_len, x1 + door_len
                area[idx] = len(masked[y1:y2, x1:x2].nonzero()[0])
            for i, j in [[0, 1], [2, 3]]:
                if area[i] > area[j]:
                    idx = i
                else:
                    idx = j
                p0 = idx2p[idx]
                p1 = (p0[0], p0[1] + door_len)
                p2 = (p0[0] + door_len, p0[1] + door_len)
                p3 = (p0[0] + door_len, p0[1])
                y1, x1 = p0
                y2, x2 = y1 + door_len, x1 + door_len
                patch = masked[y1:y2, x1:x2]
                edge = find_shaft(patch, p0, p1, p2, p3)
                edges.append(edge)
                bboxs.append([x1, y1, door_len, door_len])
        return edges, bboxs

    @classmethod
    def min_dist_to_walls(cls, p, walls, ignore=[]):
        min_d = 1e6
        close_pt = None
        x = np.array(p)
        for i, wall in enumerate(walls):
            if i in ignore:
                continue
            p1, p2, _ = wall
            pt_lst = [p1, p2]
            u = np.array(p1)
            v = np.array(p2)
            vu = v - u
            vu_norm = np.linalg.norm(vu)
            if vu_norm > 0:
                n = vu / vu_norm
                proj = u + n * np.dot(x - u, n)
                proj = (int(proj[0]), int(proj[1]))
                pt_lst.append(proj)
            for pt in pt_lst:
                d = cls.dist_points(p, pt)
                if d < min_d:
                    min_d = d
                    close_pt = pt
                    if min_d <= 0:
                        return 0, None
        return min_d, close_pt

    @classmethod
    def global_rectify(cls, doors, walls):
        def anti(direction):
            if direction == 'C':
                return 'AC'
            elif direction == 'AC':
                return 'C'
            else:
                raise ValueError('direction({}) can only be "C" or "AC"'.format(direction))

        def min_dist_to_pts(p, pts):
            min_d = 1e6
            i, j = -1, -1
            for pt, pt_i, pt_j in pts:
                d = cls.dist_points(p, pt)
                if d < min_d:
                    min_d = d
                    i = pt_i
                    j = pt_j
            return min_d, i, j


        pts = []
        for i, wall in enumerate(walls):
            p1, p2, _ = wall
            pts.extend([(p1, i, 0), (p2, i, 1)])
        short_walls = []
        for i, door in enumerate(doors):
            p1, p2, direction = door
            door_len = cls.dist_points(p1, p2)
            if direction == 'C':
                rp2 = image_utils.rotate_point(p1, p2, -90)
            elif direction == 'AC':
                rp2 = image_utils.rotate_point(p1, p2, 90)
            else:
                raise ValueError('direction({}) can only be "C" or "AC"'.format(direction))
            rp2 = [int(rp2[0]), int(rp2[1])]
            # attach to p2
            d1, d1_i, d1_j = min_dist_to_pts(p2, pts)
            d2, d2_i, d2_j = min_dist_to_pts(rp2, pts)
            if d2 < d1:
                thick = walls[d2_i][-1]
                doors[i] = [p1, rp2, anti(direction), thick]
                if 0 < d2 < door_len:
                    short_walls.append([walls[d2_i][d2_j], rp2, thick])
            else:
                thick = walls[d1_i][-1]
                doors[i] = [p1, p2, direction, thick]
                if 0 < d1 < door_len:
                    short_walls.append([walls[d1_i][d1_j], p2, thick])
            # attach to p1
            d, d_i, d_j = min_dist_to_pts(p1, pts)
            if 0 < d < door_len:
                short_walls.append([walls[d_i][d_j], p1, thick])
            # d, close_pt = cls.min_dist_to_walls(p1, walls)
            # if 0 < d < door_len and close_pt is not None:
            #     short_walls.append([close_pt, p1, thick])
        walls.extend(short_walls)
        return doors, walls

    @staticmethod
    def inside_constrain(edges, walls):
        # to constrain railing not to be surrounded by walls
        rf_edges = []
        w, h = walls.shape
        for i, (p1, p2, _) in enumerate(edges):
            x1, y1 = p1
            x2, y2 = p2
            x = x1
            while x > 0 and walls[x, y1] == 0:
                x -= 1
            if x != 0:
                continue
            y = y1
            while y > 0 and walls[x1, y] == 0:
                y -= 1
            if y != 0:
                continue
            x = x2
            while x < w and walls[x, y2] == 0:
                x += 1
            if x != w:
                continue
            y = y2
            while y < h and walls[x2, y] == 0:
                y += 1
            if y != h:
                continue
            rf_edges.append(edges[i])
        return rf_edges

    @staticmethod
    def rescale(edges, scale):

        def rescale_point(p, scale):
            return (int(p[0]*scale), int(p[1]*scale))

        for i, edge in enumerate(edges):
            p1, p2, thick = edge
            p1 = rescale_point(p1, scale)
            p2 = rescale_point(p2, scale)
            edges[i] = [p1, p2, int(thick*scale)]
        return edges

    @staticmethod
    def dump2json(ofpath, skels):

        def default(o):
            if isinstance(o, np.int64):
                return int(o)
            return o

        with open(ofpath, 'w') as of:
            print('dump skels to', ofpath)
            json.dump(skels, of, sort_keys=True, indent=4, default=default)


""" Parse text
"""
class Text(object):

    USE_OCR_SPACE = 1
    USE_OCR_GOOGLE = 0
    USE_TEXT_TENCENT = 0
    DEBUG = 0

    def __init__(self, assets_dir):
        # init templates
        self.tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/arrow.png'))
        self.h_tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/hstar.png'))
        self.v_tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/vstar.png'))
        self.up_tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/up.png'))
        self.down_tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/down.png'))
        self.left_tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/left.png'))
        self.right_tpl = mtpl_utils.create_template(os.path.join(assets_dir, 'assets/match_tpl/right.png'))
        assert self.USE_OCR_SPACE or self.USE_OCR_GOOGLE
        if self.DEBUG:
            print('Text DEBUG mode is ON')

    def extract(self, img, skel, fpath):
        # get the min-max nonzero coord
        skel = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
        xs, ys = skel.nonzero()
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        if self.DEBUG:
            print('non-zero area:', x1, x2, y1, y2)

        grid_file = ["{}_{}.png".format(fpath, x) \
                        for x in range(0, 3)]

        # body = np.ones_like(img) * 255
        # body[y1:y2, x1:x2, :] = img[y1:y2, x1:x2, :]
        text = img.copy()
        # text[y1:y2, x1:x2, :] = 255
        text[x1:x2, y1:y2, :] = 255
        image_utils.debug_save(text, grid_file[1])
        # image_utils.debug_save(body, grid_file[0])
        # rtext = image_utils.rotate(text, -90)
        # image_utils.debug_save(rtext, grid_file[2])

        img_h, img_w, _ = text.shape
        if self.DEBUG:
            tmp_img = img.copy()

        res_ocr_space = []
        res_ocr_google = []
        rooms = []

        if self.USE_OCR_SPACE:
          # with Timer('calling api of ocr space'):
            # r1 = [text_utils.ocr_space_file(f) for f in grid_file[1:]]
            r1 = [text_utils.ocr_space_file(grid_file[1])]
            r1 = text_utils.parse_ocr_space(r1)
            if len(r1) == 2:
                for r in r1[1]:
                    n, x, y, h, w = r
                    x, y = image_utils.rotate_point((0, 0), (x, y), 90)
                    x, y = int(x + img_w), int(y)
                    r1[0].append([n, x-h, y, w, h])
            res_ocr_space = r1[0]

        if self.USE_OCR_GOOGLE:
          # with Timer('calling api of vision google'):
            # r1 = [text_utils.ocr_google_file(f) for f in grid_file[1:]]
          r1 = [text_utils.ocr_google_file(grid_file[1])]
          r1 = text_utils.parse_ocr_google(r1)
          if len(r1) == 2:
              for r in r1[1]:
                  n, x, y, h, w = r
                  x, y = image_utils.rotate_point((0, 0), (x, y), 90)
                  x, y = int(x + img_w), int(y)
                  r1[0].append([n, x-h, y, w, h])
            # comment the line below to see the result without merged
          res_ocr_google = text_utils.merge_ocr_result(r1[0])

        res_number = res_ocr_space + res_ocr_google
        res_number = text_utils.merge_ocr_result(res_number)
        for r in res_number:
            self.mask_text(text, r)
        if self.DEBUG:
            print(res_number)
            for r in res_number:
                n, x, y, h, w = r
                image_utils.debug_draw_rect(tmp_img, (x, y), (x+w, y+h))
                if isinstance(n, str):
                    continue
                image_utils.debug_draw_text(tmp_img, str(n), (x, y))

        if self.USE_TEXT_TENCENT:
          # with Timer('calling api of youtu tencent'):
            r2 = [text_utils.ocr_youtu_file(f) for f in grid_file[:1]]
            r2 = text_utils.parse_ocr_youtu(r2)
            rooms = r2[0]
            if self.DEBUG:
                print(r2)
                for r in rooms:
                    n, x, y, h, w = r
                    image_utils.debug_draw_rect(tmp_img, (x, y), (x+w, y+h))
                    image_utils.debug_draw_text(tmp_img, str(n), (x+w, y), chinese=True)

        # with Timer('template match'):
        tpls = []
        # match compass
        r = mtpl_utils.match_template(text, self.tpl)
        x, y, h, w = r
        compass = self.compass_direction(text[y:y+h, x:x+w])
        if self.DEBUG:
            image_utils.debug_draw_rect(tmp_img, (x, y), (x+w, y+h))

        # match anchor (hstar, left, right; vstar, up, down)
        scales = []
        hps, vps = text_utils.analyze_panel_by_bbox(res_number, img_w, img_h)

        for panel in hps:
            lst, px, py, ph, pw = panel
            rs = mtpl_utils.match_multi_template(text[py:py+ph, px:px+pw, :], self.h_tpl, num=len(lst)-1, auto=True)
            for i, r in enumerate(rs):
                x, y, h, w = r
                rs[i] = ('hstar', x+px, y+py, h, w)
                self.mask_text(text, rs[i])
            r = mtpl_utils.match_template(text[py:py+ph, px:px+pw], self.left_tpl, auto=True)
            if r is not None:
                x, y, h, w = r
                r = ('left', x+px, y+py, h, w)
                self.mask_text(text, r)
                rs.append(r)
            r = mtpl_utils.match_template(text[py:py+ph, px:px+pw], self.right_tpl, auto=True)
            if r is not None:
                x, y, h, w = r
                r = ('right', x+px, y+py, h, w)
                self.mask_text(text, r)
                rs.append(r)
            scales.extend(self.estimate_scale(lst, rs, axis=1))
            tpls.extend(rs)
            if self.DEBUG:
                for r in rs:
                    _, x, y, h, w = r
                    image_utils.debug_draw_rect(tmp_img, (x, y), (x+w, y+h))
        for panel in vps:
            lst, px, py, ph, pw = panel
            rs = mtpl_utils.match_multi_template(text[py:py+ph, px:px+pw, :], self.v_tpl, num=len(lst)-1, auto=True)
            for i, r in enumerate(rs):
                x, y, h, w = r
                rs[i] = ('vstar', x+px, y+py, h, w)
                self.mask_text(text, rs[i])
            r = mtpl_utils.match_template(text[py:py+ph, px:px+pw], self.up_tpl, auto=True)
            if r is not None:
                x, y, h, w = r
                r = ('up', x+px, y+py, h, w)
                self.mask_text(text, r)
                rs.append(r)
            r = mtpl_utils.match_template(text[py:py+ph, px:px+pw], self.down_tpl, auto=True)
            if r is not None:
                x, y, h, w = r
                r = ('down', x+px, y+py, h, w)
                self.mask_text(text, r)
                rs.append(r)
            scales.extend(self.estimate_scale(lst, rs, axis=2))
            tpls.extend(rs)
            if self.DEBUG:
                for r in rs:
                    _, x, y, h, w = r
                    image_utils.debug_draw_rect(tmp_img, (x, y), (x+w, y+h))
        # store the text detection result to json
        self.dump2json(fpath+'.json', compass, res_number, tpls, rooms, scales)
        scales = np.array(scales)
        if self.DEBUG:
            print(len(scales), scales, np.median(scales))
        return np.median(scales)

    @staticmethod
    def mask_text(text, r):
        _, x, y, h, w = r
        text[y:y+h, x:x+w, :] = 255

    @staticmethod
    def compass_direction(img):
        # 1, 2, 3, 4 is clock-wise
        gray = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        x, y, w, h = cv2.boundingRect(c)
        gray = gray[y:y+h, x:x+w]
        h, w = gray.shape
        mh, mw = int(h / 2), int(w / 2)
        p1 = len(gray[:mh, :mw].nonzero()[0])
        p2 = len(gray[:mh, mw:].nonzero()[0])
        p3 = len(gray[mh:, mw:].nonzero()[0])
        p4 = len(gray[mh:, :mw].nonzero()[0])
        th = (p1 + p2 + p3 + p4) * 0.05
        if abs(p1 + p2 - p3 - p4) < th:
            head1 = 'median'
        elif p1 + p2 < p3 + p4:
            head1 = 'up'
        else:
            head1 = 'down'
        if abs(p1 + p4 - p2 - p3) < th:
            head2 = 'median'
        elif p1 + p4 < p2 + p3:
            head2 = 'left'
        else:
            head2 = 'right'
        compass = '{}-{}'.format(head1, head2)
        # cv2.imwrite('compass.png', gray)
        # print("number of pixels in 4 parts: {}, {}, {}, {}. (th: {})".\
        #         format(p1, p2, p3, p4, th))
        print("compass: {}".format(compass))
        return compass

    @staticmethod
    def estimate_scale(lst, rs, axis):
        # axis=1,2 means horizontal and vertical respectively
        rs = sorted(rs, key=lambda x:x[axis])
        idx = 0
        size = len(lst)
        scales = []
        for i in range(len(rs) - 1):
            c1, len1 = rs[i][axis], rs[i][-axis]
            c2, len2 = rs[i+1][axis], rs[i+1][-axis]
            t1 = lst[idx][axis]
            while t1 < c1 and idx < size - 1:
                idx += 1
            if t1 < c2:
                n = lst[idx][0]
                pixel_dist = c2 + len2 / 2 - (c1 + len1 / 2)
                if pixel_dist == 0:
                    print(c1, c2, len1, len2)
                    continue
                scale = 1. * n / pixel_dist
                scales += [scale]
        return scales

    @staticmethod
    def dump2json(ofpath, compass, res_number, tpls, rooms, scales):
        info = {
            'compass': compass,
            'number': res_number,
            'templates': tpls,
            'rooms': rooms,
            'scales': scales,
        }
        with open(ofpath, 'w') as of:
            print('dump text to', ofpath)
            json.dump(info, of, sort_keys=True, indent=4)

    @staticmethod
    def loadjson(fpath):
        with open(fpath) as f:
            print('load text from', fpath)
            info = json.load(f)
            scales = info['scales']
            scales = np.array(scales)
            return np.median(scales)


# if __name__ == '__main__':
def extract():
    program_dir = '/home/chr/sgy/web/program/'
    work_dir = os.path.join(program_dir, 'assets', 'datasets')
    segment_folder = os.path.join(work_dir, 'segmentation_results')
    assets_dir = os.path.join(program_dir, 'seg2skel')
    min_size = 500

    photos = glob.glob(os.path.join(segment_folder, '*_image.png'))
    text = Text(assets_dir)
    skel = Skeleton()
    for fpath in photos:

        # example: fpath == 'segmentation_results/a_image.png'
        print('process ', fpath)
        ori_img = image_utils.imread(fpath)
        if ori_img.shape[0] < min_size and ori_img.shape[1] < min_size:
            continue

        # seg_img == 'a_prediction.png'
        seg_img = image_utils.imread(fpath[:-9] + 'prediction.png')

        # output _text.json, according to origin image and (already generated) segmentation image

        # ofpath == 'a_image_text_json'
        ofpath = fpath[:-4] + '_text.json'
        if not os.path.isfile(ofpath):
            scale = text.extract(ori_img, seg_img, ofpath[:-5])
        else:
            scale = text.loadjson(ofpath)
        # print('[[[[[[[[[[[[[')
        # print(scale)
        # print(']]]]]]]]]]]]]')
        # input()

        ofpath = fpath[:-4] + '_skels.json'
        skel.extract(ori_img, seg_img, ofpath, scale)
        if not os.path.isfile(ofpath):
            skel.extract(ori_img, seg_img, ofpath, scale)

def load():
    program_dir = '/home/chr/sgy/web/program/'
    work_dir = os.path.join(program_dir, 'assets', 'datasets')
    segment_folder = os.path.join(work_dir, 'segmentation_results')

    e = 1
    photos = glob.glob(os.path.join(segment_folder, '*_skels.json'))
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
                if -e < p1.x - p2.x < e and -e < p1.y - p2.y < e:
                    continue
                if key == 'wall' or key == 'wall_bay' or key == 'railing':
                    wall = Wall(p1, p2, thick, kind=kind)
                    # print(wall.json())
                    if not wall.is_point():
                        walls.append(wall)
                    else:
                        raise Exception()
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
