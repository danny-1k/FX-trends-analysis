import numpy as np

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


class SeriesToImageConverter:
    def __init__(self, height, width):
        """
        Wrapper that makes calling `create_image_from_time_series` easier.

        Args:
            height (int): Height of Image.
            width (int): Width of Image.
        """
        
        self.height = height
        self.width = width

    def __call__(self, chunk):
        image = create_image_from_time_series(window=chunk, height=self.height, width=self.width)

        return image