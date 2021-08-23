from typing import Union, Any

import bpy
from src.utility.Utility import Utility, KeyFrame
from mathutils import Vector, Euler, Color, Matrix

from typing import List

class Struct:

    def __init__(self, object: bpy.types.Object):
        self.blender_obj = object

    def set_name(self, name: str):
        """ Sets the name of the entity.

        :param name: The new name.
        """
        self.blender_obj.name = name

    def get_name(self) -> str:
        """ Returns the name of the entity.

        :return: The name.
        """
        return self.blender_obj.name

    def get_cp(self, key: str, frame: int = None) -> Any:
        """ Returns the custom property with the given key.

        :param key: The key of the custom property.
        :param frame: The frame number at which the value should be returned. If None is given, the current frame number is used.
        :return: The value of the custom property.
        """
        with KeyFrame(frame):
            return self.blender_obj[key]

    def set_cp(self, key: str, value: Any, frame: int = None):
        """ Sets the custom property with the given key.

        Keyframes can be only set for custom properties for the types int, float or bool.

        :param key: The key of the custom property.
        :param value: The value to set.
        :param frame: The frame number which the value should be set to. If None is given, the current frame number is used.
        """
        self.blender_obj[key] = value
        if isinstance(self.blender_obj[key], float) or isinstance(self.blender_obj[key], int):
            Utility.insert_keyframe(self.blender_obj, "[\"" + key + "\"]", frame)

    def del_cp(self, key: str):
        """ Removes the custom property with the given key.

        :param key: The key of the custom property to remove.
        """
        del self.blender_obj[key]

    def has_cp(self, key: str) -> bool:
        """ Return whether a custom property with the given key exists.

        :param key: The key of the custom property to check.
        :return: True, if the custom property exists.
        """
        return key in self.blender_obj

    def get_all_cps(self) -> list:
        """ Returns all custom properties as key, value pairs.

        :return: A list of key value pairs
        """
        return self.blender_obj.items()

    def clear_all_cps(self):
        """ Removes all existing custom properties the struct has. """
        keys = self.blender_obj.keys()
        for key in keys:
            del self.blender_obj[key]

    def __setattr__(self, key, value):
        if key != "blender_obj":
            raise Exception("The API class does not allow setting any attribute. Use the corresponding method or directly access the blender attribute via entity.blender_obj.attribute_name")
        else:
            object.__setattr__(self, key, value)
