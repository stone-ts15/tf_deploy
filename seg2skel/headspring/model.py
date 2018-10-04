# -*- coding: utf-8
#

import math
import typing
import itertools
from .interface import ItemAssignable, JsonSupported, JsonSupportedEnum

""" Geometry
"""

class Point(ItemAssignable, JsonSupported):

    def __init__(self, x: typing.Union[int, float], y: typing.Union[int, float]):
        self.x = x
        self.y = y

    def __lt__(self, other: 'Point'):
        # e = cfg.MM_EPSILON
        e = 1
        if self.x < other.x - e:
            return True
        elif self.x > other.x + e:
            return False
        if self.y < other.y - e:
            return True
        elif self.y > other.y + e:
            return False
        return False

    def __gt__(self, other: 'Point'):
        # e = cfg.MM_EPSILON
        e = 1
        if self.x < other.x - e:
            return False
        elif self.x > other.x + e:
            return True
        if self.y < other.y - e:
            return False
        elif self.y > other.y + e:
            return True
        return False

    @classmethod
    def from_json(cls, json: dict):
        return cls(x=json['x'], y=json['y'])

    def json(self):
        return {
            'x': self.x, 'y': self.y,
        }


class Vector(ItemAssignable, JsonSupported):

    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def length(self):
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        return math.sqrt(dx * dx + dy * dy)

    @property
    def dx(self):
        return self.p2.x - self.p1.x

    @property
    def dy(self):
        return self.p2.y - self.p1.y

    def is_point(self):
        # e = cfg.MM_EPSILON # hack
        e = 1
        return -e < self.dx < e and -e < self.dy < e

    def direction_unknown(self):
        raise VectorDirectionUnknown()

    def assert_not_point(self):
        if self.is_point():
            self.direction_unknown()

    def get_p1_vertical_offset_point(self, offset: float):
        """
        p1向垂直方向平移一段距离，平移方向为向量方向顺时针旋转90度方向
        """
        self.assert_not_point()
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        d = math.sqrt(dx * dx + dy * dy)
        return Point(self.p1.x + offset / d * dy, self.p1.y - offset / d * dx)

    @classmethod
    def from_json(cls, json: dict):
        return cls(p1=Point.from_json(json['p1']), p2=Point.from_json(json['p2']))

    def json(self):
        return {
            'p1': self.p1.json(), 'p2': self.p2.json(),
        }


class ThickVector(Vector):

    def __init__(self, p1: Point, p2: Point, thickness: int = 0):
        super(ThickVector, self).__init__(p1, p2)
        self.thickness = thickness

    @classmethod
    def from_json(cls, json: dict):
        return cls(p1=Point.from_json(json['p1']), p2=Point.from_json(json['p2']), thickness=json.get('thickness', 0))

    def json(self):
        return {
            'p1': self.p1.json(), 'p2': self.p2.json(), 'thickness': self.thickness,
        }


""" Wall
"""

class WallType(JsonSupportedEnum):

    NON_BEARING = 0     # 非承重墙（默认）
    BEARING = 1         # 承重墙
    SOFT = 2            # 矮墙/室内软隔断
    RAILING = 3         # 室外栏杆（阳台等）
    BAY = 4             # 飘窗基底
    VIRTUAL = 5         # 非实体墙


class WallMeta(ItemAssignable, JsonSupported):

    def __init__(self,
                 p1_padding: typing.Optional[int] = None,
                 p2_padding: typing.Optional[int] = None,
                 outer: typing.Optional[bool] = None,
                 height: typing.Optional[int] = None):
        self.p1_padding = p1_padding if p1_padding is not None else 0
        self.p2_padding = p2_padding if p2_padding is not None else 0
        self.outer = outer if outer is not None else False      # 是否为外墙，房间的墙该属性无效
        self.height = height    # 对于SOFT、RAILING类型的墙有效

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            p1_padding=json.get('p1Padding'), p2_padding=json.get('p2Padding'),
            outer=json.get('outer')
        )

    def json(self):
        return {
            'p1Padding': self.p1_padding, 'p2Padding': self.p2_padding, 'outer': self.outer
        }


class Wall(ThickVector):

    def __init__(self, p1: Point, p2: Point, thickness: int,
                 kind: typing.Optional[WallType] = None,
                 meta: typing.Optional[WallMeta] = None):
        """
        定义墙的中轴位置
        墙的厚度必须显式地给出
        需要提供两个方向的延展长度（padding），以便在墙的连接点也能美观地进行绘制
        """
        super(Wall, self).__init__(p1, p2, thickness)
        self.kind = kind or WallType.default()
        self.meta = meta or WallMeta()

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            p1=Point.from_json(json['p1']), p2=Point.from_json(json['p2']), thickness=json['thickness'],
            kind=WallType.from_json(json.get('kind')), meta=WallMeta.from_json(json.get('meta', dict()))
        )

    def json(self):
        return {
            'p1': self.p1.json(), 'p2': self.p2.json(), 'thickness': self.thickness,
            'kind': self.kind.json(), 'meta': self.meta.json(),
        }


""" Window
"""

class WindowType(JsonSupportedEnum):

    COMMON = 0      # 普通窗
    FRENCH = 1      # 落地窗
    BAY_ONLY = 2    # 无窗扇的飘窗（有窗扇的飘窗属于普通窗）


class WindowOpenDirection(JsonSupportedEnum):

    WHATEVER = 0    # 两个方向均可，即落地窗/无窗扇的飘窗/无法打开的普通窗
    INWARD = 1      # 朝屋内
    OUTWARD = 2     # 朝屋外


class WindowHandlePosition(JsonSupportedEnum):

    NOT_EXIST = 0   # 无窗把手，必须保证开窗方向为 WHATEVER
    TOP = 1         # 把手在窗户上方
    LEFT = 2        # 在窗户左方
    BOTTOM = 3      # 在下方
    RIGHT = 4       # 在右方


class WindowMeta(JsonSupported, ItemAssignable):

    def __init__(self,
                 bay_wall: typing.Optional[Wall] = None,
                 outer_vector: typing.Optional[Vector] = None,
                 thickness: typing.Optional[int] = None):
        self.bay_wall = bay_wall
        self.outer_vector = outer_vector
        self.thickness = thickness

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            bay_wall=Wall.from_json_or_none(json.get('bayWall')),
            outer_vector=Vector.from_json_or_none(json.get('outerVector')),
            thickness=json.get('thickness'),
        )

    def json(self):
        return {
            'bayWall': self.json_or_none(self.bay_wall),
            'outerVector': self.json_or_none(self.outer_vector),
            'thickness': self.thickness,
        }


class Window(Vector):

    def __init__(self, p1: Point, p2: Point, meta: typing.Optional[WindowMeta] = None,
                 kind: typing.Optional[WindowType] = None,
                 open_direction: typing.Optional[WindowOpenDirection] = None,
                 handle_position: typing.Optional[WindowHandlePosition] = None,
                 height: typing.Optional[int] = None, sill_height: typing.Optional[int] = None,
                 bay_depth: typing.Optional[int] = None,
                 wall: typing.Optional[Wall] = None):
        super(Window, self).__init__(p1, p2)
        self.kind = kind or WindowType.default()
        self.open_direction = open_direction or WindowOpenDirection.default()
        self.handle_position = handle_position or WindowHandlePosition.default()
        self.height = height if height is not None else 1500
        self.sill_height = sill_height if sill_height is not None else 1200
        self.bay_depth = bay_depth if bay_depth is not None else 0      # 飘窗的窗台在 p1->p2 方向的右侧
        self.meta = meta or WindowMeta()
        self.wall = wall

    @classmethod
    def from_json(cls, json: dict):
        # 忽略 wall
        return cls(
            p1=Point.from_json(json['p1']), p2=Point.from_json(json['p2']),
            kind=WindowType.from_json(json.get('kind')),
            open_direction=WindowOpenDirection.from_json(json.get('openDirection')),
            handle_position=WindowHandlePosition.from_json(json.get('handlePosition')),
            height=json.get('height'), sill_height=json.get('sillHeight'), bay_depth=json.get('bayDepth'),
            meta=WindowMeta.from_json(json.get('meta', dict())),
        )

    def json(self):
        # 忽略 wall
        return {
            'p1': self.p1.json(), 'p2': self.p2.json(),
            'kind': self.kind.json(), 'openDirection': self.open_direction.json(),
            'handlePosition': self.handle_position.json(), 'height': self.height, 'sillHeight': self.sill_height,
            'bayDepth': self.bay_depth,
            'meta': self.meta.json(),
        }


""" Door
"""

class DoorType(JsonSupportedEnum):

    SINGLE = 0              # 单开门（默认）
    SLIDING = 1             # 推拉门
    EQUAL_DOUBLE = 2        # 双开门
    UNEQUAL_DOUBLE = 3      # 子母门
    OPENING = 4             # 门洞


class DoorOpenDirection(JsonSupportedEnum):

    WHATEVER = 0            # 两个方向均可，即推拉门或门洞
    CLOCKWISE = 1           # 顺时针
    ANTI_CLOCKWISE = 2      # 逆时针


class DoorMeta(ItemAssignable, JsonSupported):

    # 入户门属性相关
    NOT_ENTRANCE = 0    # 非入户门
    ENTRANCE = 1        # 入户门
    MAIN_ENTRANCE = 2   # 主入户门

    def __init__(self,
                 entrance: typing.Optional[int] = None,
                 thickness: typing.Optional[int] = None,
                 p1_leaf_point: typing.Optional[Point] = None,
                 related_rooms: typing.Optional[typing.List] = None):
        self.entrance = int(entrance) if entrance is not None else 0
        self.thickness = thickness
        self.p1_leaf_point = p1_leaf_point
        self.related_rooms = related_rooms or list()

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            entrance=json.get('entrance'),
            thickness=json.get('thickness'),
            p1_leaf_point=Point.from_json_or_none(json.get('p1LeafPoint')),
            related_rooms=json.get('relatedRooms'),
        )

    def json(self):
        return {
            'entrance': self.entrance,
            'thickness': self.thickness,
            'p1LeafPoint': self.json_or_none(self.p1_leaf_point),
            'relatedRooms': self.related_rooms,
        }


class Door(Vector):

    def __init__(self, p1: Point, p2: Point,
                 kind: typing.Optional[DoorType] = None,
                 open_direction: typing.Optional[DoorOpenDirection] = None,
                 height: typing.Optional[int] = None, sill_height: typing.Optional[int] = None,
                 meta: typing.Optional[DoorMeta] = None,
                 wall: typing.Optional[Wall] = None):
        super(Door, self).__init__(p1, p2)
        self.kind = kind or DoorType.default()
        self.open_direction = open_direction or DoorOpenDirection.default()
        self.height = height if height is not None else 2000
        self.sill_height = sill_height if sill_height is not None else 0
        self.meta = meta or DoorMeta()
        self.wall = wall
        self.update_meta()

    def update_meta(self):
        if self.open_direction == DoorOpenDirection.CLOCKWISE:
            self.meta.p1_leaf_point = self.get_p1_vertical_offset_point(self.length())
        elif self.open_direction == DoorOpenDirection.ANTI_CLOCKWISE:
            self.meta.p1_leaf_point = self.get_p1_vertical_offset_point(-self.length())
        else:
            self.meta.p1_leaf_point = None

    @classmethod
    def from_json(cls, json: dict):
        # 从 JSON 构建的，wall 保持为 None
        return cls(
            p1=Point.from_json(json['p1']), p2=Point.from_json(json['p2']),
            kind=DoorType.from_json(json.get('kind')),
            open_direction=DoorOpenDirection.from_json(json.get('openDirection')),
            height=json.get('height'), sill_height=json.get('sillHeight'),
            meta=DoorMeta.from_json(json.get('meta', dict())),
        )

    def json(self):
        # 忽略 wall
        return {
            'p1': self.p1.json(), 'p2': self.p2.json(),
            'kind': self.kind.json(), 'openDirection': self.open_direction.json(),
            'height': self.height, 'sillHeight': self.sill_height,
            'meta': self.meta.json(),
        }


""" House
"""

class HouseRequirement(JsonSupported, ItemAssignable):

    def __init__(self):
        pass

    def deepcopy(self):
        return self.__class__()

    @classmethod
    def from_json(cls, json: dict):
        return cls()

    def json(self):
        return {}


class HouseMeta(JsonSupported, ItemAssignable):

    def __init__(self,
                 outer_walls: typing.Optional[typing.List[Wall]] = None,
                 outer_surfaces = None, # hack
                 corner_surfaces = None, # hack
                 # outer_surfaces: typing.Optional[typing.List[Surface]] = None,
                 # corner_surfaces: typing.Optional[typing.List[Surface]] = None,
                 name: typing.Optional[str] = None,
                 height: typing.Optional[int] = None,
                 default_perspective: typing.Optional[Vector] = None):
        # 外边缘墙
        self.outer_walls = outer_walls or list()    # type: typing.List[Wall]
        # 户型外边缘
        self.outer_surfaces = outer_surfaces or list()      # type: typing.List[Surface]
        # 外墙L转角附加面
        self.corner_surfaces = corner_surfaces or list()    # type: typing.List[Surface]
        # 户型名
        self.name = name
        # 层高
        self.height = height or 2700
        # 默认视角
        self.default_perspective = default_perspective

    @classmethod
    def from_json(cls, json: dict):
        return cls(
            outer_walls=[Wall.from_json(inst) for inst in json.get('outerWalls', list())],
            outer_surfaces=[Surface.from_json(inst) for inst in json.get('outerSurfaces', list())],
            corner_surfaces=[Surface.from_json(inst) for inst in json.get('cornerSurfaces', list())],
            name=json.get('name'),
            height=json.get('height'),
            default_perspective=Vector.from_json_or_none(json.get('defaultPerspective')),
        )

    def json(self):
        return {
            'outerWalls': [inst.json() for inst in self.outer_walls],
            'outerSurfaces': [inst.json() for inst in self.outer_surfaces],
            'cornerSurfaces': [inst.json() for inst in self.corner_surfaces],
            'name': self.name,
            'height': self.height,
            'defaultPerspective': self.json_or_none(self.default_perspective),
        }


class House(JsonSupported, ItemAssignable):

    VERSION = 2

    def __init__(self,
                 walls: typing.List[Wall],
                 doors: typing.List[Door],
                 windows: typing.List[Window],
                 cubic_columns = None, # hack
                 rooms = None,
                 # cubic_columns: typing.List[CubicColumn],
                 # rooms: typing.Optional[RoomList] = None,
                 rotation: typing.Optional[float] = None,
                 requirement: typing.Optional[HouseRequirement] = None,
                 meta: typing.Optional[HouseMeta] = None,
                 template_filters = None):
                 # template_filters: typing.Optional[TemplateFilterList] = None):
        self.walls = walls
        self.doors = doors
        self.windows = windows
        self.cubic_columns = cubic_columns or list() # hack
        self.rooms = rooms or list()        # type: RoomList
        self.rotation = rotation
        self.requirement = requirement or HouseRequirement()
        self.meta = meta or HouseMeta()
        self.template_filters = template_filters or list()      # type: TemplateFilterList
        # self.normalize()

    @property
    def height(self):
        return self.meta.height

    def json(self):
        return {
            'version': self.VERSION,
            'walls': self.jsonify_list(self.walls),
            'doors': self.jsonify_list(self.doors),
            'windows': self.jsonify_list(self.windows),
            'cubicColumns': self.jsonify_list(self.cubic_columns),
            'rooms': self.jsonify_list(self.rooms),
            'rotation': self.rotation,
            'requirement': self.requirement.json(),
            'meta': self.meta.json(),
            'templateFilters': self.jsonify_list(self.template_filters),
        }
