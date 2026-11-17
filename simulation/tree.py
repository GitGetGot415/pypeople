from dataclasses import dataclass, field
from constants import STARTING_TREE_POSITION


@dataclass
class Tree:
    x: float = STARTING_TREE_POSITION[0]
    y: float = STARTING_TREE_POSITION[1]
    id: str = "fruit-tree"
    fruit: int = 0
    shake_remaining: float = field(default=0.0, repr=False, compare=False)

    @property
    def interaction_point(self):
        return (self.x - 60.0, self.y + 50.0)
