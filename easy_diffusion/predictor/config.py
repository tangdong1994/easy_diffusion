from enum import Enum

class Direction(Enum):
    """
    图像的移动方向
    pic move direction
    center表示图像放大
    center represent zoom in
    """
    up = 1
    down = 2
    left = 3
    right = 4
    center = 5
    left_up = 6
    left_down = 7
    right_up = 8
    right_down = 9
    circle = 10