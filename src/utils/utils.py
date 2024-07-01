import numpy as np


class RunningAverager:
    """
    Stores the running average of a value over time by smoothening / Interpolation.
    v_{t+1} = α * v_t + (1-α) * v_{t-1}
    """
    def __init__(self, smooth:float=.7) -> None:
        self.value = 0
        self.smooth = smooth

    def update(self, value:float) -> None:
        """
        Update the value stored through interpolation.
        """
        self.value = self.smooth * self.value + (1-self.smooth) * value

    def reset(self) -> None:
        """
        Restarts the averager by setting the value to 0.
        """
        self.value = 0