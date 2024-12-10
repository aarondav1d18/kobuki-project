from dataclasses import dataclass
from enum import Enum
from typing import List
from std_msgs.msg import Float64MultiArray
from .Pos import Point

@dataclass
class CurrentPosition:
    ## idk why the fuck this was made in original but hey ho
    ## dont think this is actually every used but i dont give a fuck and cant be asked to change the
    ## perception cone
    pos: Point

    def __repr__(self) -> str:
        # return "(%f, %f, %s)" % (self.x, self.y, self.colour)
        return (str(self.cone))
