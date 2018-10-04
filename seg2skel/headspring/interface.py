# -*- coding: utf-8
#
from enum import Enum
import typing


class ItemAssignable:

    def __getitem__(self, item):
        return vars(self)[item]

    def __setitem__(self, key, value):
        if key in vars(self):
            setattr(self, key, value)
        else:
            raise TypeError('Attribute \'{}\' does not support item assignment'.format(key))


class JsonSerializable:

    def json(self):
        raise NotImplementedError('Sub-class of JsonSerializable should implement the json() method')

    @classmethod
    def json_or_none(cls, obj: typing.Optional['JsonSerializable']):
        if obj is None:
            return None
        return obj.json()

    @classmethod
    def jsonify_list(cls, lst: typing.List['JsonSerializable']):
        return [inst.json() for inst in lst]


class JsonDeserializable:

    @classmethod
    def from_json(cls, json: dict):
        raise NotImplementedError('Sub-class of JsonDeserializable should implement the from_json() method')

    @classmethod
    def from_json_or_none(cls, json: typing.Optional[dict]):
        if json is None:
            return None
        return cls.from_json(json)

    @classmethod
    def from_json_array(cls, arr: typing.Optional[list]):
        if arr is None:
            return list()
        return [cls.from_json(inst) for inst in arr]


# noinspection PyAbstractClass
class JsonSupported(JsonSerializable, JsonDeserializable):
    """ 支持JSON序列化和反序列化的接口 """


class JsonSupportedEnum(JsonSupported, Enum):

    def json(self):
        return self.name.upper().replace('-', '_')

    @classmethod
    def from_json(cls, json: typing.Optional[str]):
        if json is None:
            return cls.default()
        name = json.upper().replace('-', '_')
        for inst in cls:
            if inst.json() == name:
                return inst
        raise TypeError('Unknown {}: {}'.format(cls.__name__, json))

    @classmethod
    def default(cls):
        # noinspection PyArgumentList
        return cls(0)   # 默认值为0
