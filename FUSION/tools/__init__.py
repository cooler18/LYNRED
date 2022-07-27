# __all__ = ["gradient_tools.py", "data_management_tools", "manipulation_tools"]
# from .tool_gradients import *
# from .data_management_tools import *

from FUSION.classes.Image import ImageCustom
__domain__ = {
    'HSV':      [ImageCustom.HSV, 2],
    'LAB':      [ImageCustom.LAB, 0],
    'HLS':      [ImageCustom.HLS, 1],
    'LUV':      [ImageCustom.LUV, 0],
    'YCrCb':    [ImageCustom.YCrCb, 0]
            }
