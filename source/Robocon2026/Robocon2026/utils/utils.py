import numpy as np
from scipy.spatial.transform import Rotation as R


def euler2quaternion(euler):
    r = R.from_euler("xyz", euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion
