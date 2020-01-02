from airobot.ee_tool.ee import EndEffectorTool


class AllegroHandReal(EndEffectorTool):
    def __init__(self, cfgs):
        """
        Constructor for Allegro Hand class

        Args:
            cfgs (YACS CfgNode): configurations for the hand
        """
        super(AllegroHandReal, self).__init__(cfgs=cfgs)
