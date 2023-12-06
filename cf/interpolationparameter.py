import cfdm

from . import mixin


class InterpolationParameter(
    mixin.PropertiesData, cfdm.InterpolationParameter
):
    pass
