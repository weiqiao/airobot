from airobot.cfgs.allegro_hand import get_allegro_cfg
from airobot.cfgs.default_configs import get_cfg_defaults

_C = get_cfg_defaults()
# whether the robot has an arm or not
_C.HAS_ARM = False
# whether the robot has a end effector tool or not
_C.HAS_EETOOL = True

_C.ROBOT_DESCRIPTION = '/robot_description'

# prefix of the class name of the ARM
# if it's for pybullet simulation, the name will
# be augemented to be '<Prefix>Pybullet'
# if it's for the real robot, the name will be
# augmented to be '<Prefix>Real'
_C.ARM.CLASS = 'ARM'

_C.EETOOL = get_allegro_cfg()
_C.EETOOL.CLASS = 'AllegroHand'


def get_cfg():
    return _C.clone()
