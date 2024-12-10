from dataclasses import dataclass
from enum import Enum
from typing import List
from std_msgs.msg import Float64MultiArray

@dataclass
class ConeMap:
    ## idk why the fuck this was made in original but hey ho
    ## dont think this is actually every used but i dont give a fuck and cant be asked to change the
    ## perception cone
    cone: Float64MultiArray

    def __repr__(self) -> str:
        # return "(%f, %f, %s)" % (self.x, self.y, self.colour)
        return (str(self.cone))
