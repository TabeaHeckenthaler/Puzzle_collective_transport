import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter


def smooth_array(array, fps, kernel_size=None):
    if kernel_size is None:
        kernel_size = 8 * (fps // 4) + 1
    new_array = medfilt(array, kernel_size=kernel_size)
    new_array = gaussian_filter(new_array, sigma=kernel_size // 5)
    return new_array


def flatten(t: list):
    if isinstance(t[0], list):
        return [item for sublist in t for item in sublist]
    else:
        return t


def ranges(nums, *args, **kwargs):
    if 'scale' in kwargs:
        scale = kwargs['scale']
    else:
        scale = np.linspace(0, len(nums) - 1, len(nums) - 0)

    if 'boolean' in args:
        nums = np.where(np.array(nums))[0]

    if 'smallest_gap' in kwargs:
        smallest_gap = kwargs['smallest_gap']
    else:
        smallest_gap = 2

    if 'buffer' in kwargs:
        buffer = kwargs['buffer']
    else:
        buffer = 0

    if len(nums) == 0:
        return []

    if len(nums) == 1:
        return [[scale[nums[0]], scale[nums[0] + 1]]]

    ran = [[scale[nums[0]], scale[nums[-1]]]]
    for i in range(len(nums) - 1):
        if smallest_gap < nums[i + 1] - nums[i]:
            ran[-1] = [ran[-1][0], scale[nums[i]] + 1 + buffer]
            ran.append([scale[nums[i + 1]] - buffer, scale[nums[-1]] + buffer])

    return ran