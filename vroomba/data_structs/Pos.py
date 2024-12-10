from dataclasses import dataclass
from enum import Enum
from typing import List
from std_msgs.msg import Float64MultiArray

@dataclass
class Point:
    x: float
    y: float
    angle: float

    def __repr__(self) -> str:
        # return "(%f, %f, %s)" % (self.x, self.y, self.colour)
        return "(%f, %f)" % (self.x, self.y)

