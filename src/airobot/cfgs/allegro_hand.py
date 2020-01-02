from yacs.config import CfgNode as CN

_C = CN()


def get_allegro_cfg():
    return _C.clone()
