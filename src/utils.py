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


def create_image_from_time_series(window, height=350, width=500):
    """
    Generates a binary image of shape (height, width) from the given
    time series window representing the trend in a specific time frame.

    Regardless of the number of samples in the window, the function returns a fixed-size
    image to mimick the pattern a person would see looking at the graph.

    The x and y values are interpolated to the width and height specified respectively
    and those are normalised and scaled to obtain indices for the binary image matrix.

    Args:
        window (np.ndarray): Time-series window.
        height (int): Height of Image.
        width (int): Width of Image.
    """


    # first interpolate prices from T = t ... T = width

    x = np.array(list(range(1, window.shape[0] + 1)))
    xvals = np.linspace(1, window.shape[0], width)
    # print(xvals.shape, x.shape, window.shape)
    yinterp = np.interp(xvals, x, window)


    # now, the x-axis is as long as the width specified.
    # next, we need to stretch the y axis so that we can fill the image

    image = np.zeros((height, width))

    ymax = yinterp.max()
    ymin = yinterp.min()

    if ymax - ymin == 0:
        return image

    y_normalised = (yinterp - ymin) / (ymax - ymin)

    y = (y_normalised * height)


    for i in range(width):
        j = y[i]
        j = int(j+.5)
        image[height - j - 1, i] = 1

    return image