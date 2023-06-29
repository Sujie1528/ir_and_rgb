import math
import os
import cv2
import numpy as np


cur_dir = os.path.dirname(__file__)

_colormap_names = ["ironbow", "white_hot", "black_hot", "rainbow", "rainbow_hc", "outdoor_alert", "arctic", "turbo"]
_colormaps = [os.path.join(cur_dir, "colormap/{}.npy".format(n)) for n in _colormap_names]
COLORMAPS = {n: np.flip(np.load(p), axis=1) for n, p in zip(_colormap_names, _colormaps)}


class Circle:
    def __init__(self, cx, cy, radius, thickness=1, size=None):
        self.cx = cx
        self.cy = cy
        self.r = radius
        self.t = thickness
        self.size = size
        self.refresh()
            
    def refresh(self):
        x1 = max(math.ceil(self.cx - self.r - self.t), 0)
        y1 = max(math.ceil(self.cy - self.r - self.t), 0)
        x2 = math.floor(self.cx + self.r + self.t) + 1
        y2 = math.floor(self.cy + self.r + self.t) + 1
        
        if self.size is not None:
            h, w = self.size
            x2 = min(x2, w)
            y2 = min(y2, h)
            
        vs, us = np.meshgrid(range(y1, y2), range(x1, x2), indexing="ij")
        d2 = (vs - self.cy) ** 2 + (us - self.cx) ** 2
        mask = (d2 >= self.r ** 2) & (d2 <= (self.r + self.t) ** 2) | (d2 <= 1.4 ** 2)
        self.ys = vs[mask]
        self.xs = us[mask]
        
        
def pseudo_color(gray_norm, colormap):
    """
    gray_norm, 2D np.ndarray or 1D np.ndarray
    """
    #gray_norm = gray_norm.clip(min=0.0, max=1.0)
    position = gray_norm * (colormap.shape[0] - 1)
    
    down = np.floor(position).astype(np.int32)
    up = np.ceil(position).astype(np.int32)
    r = np.expand_dims(position - down, -1)
    
    image = r * colormap[up] + (1 - r) * colormap[down]
    return image.astype(np.uint8)


def post_bilinear(matrix, new_size, xy):
    ih, iw = matrix.shape
    nh, nw = new_size
    x, y = xy

    cx = iw / nw * (x + 0.5) - 0.5
    cy = ih / nh * (y + 0.5) - 0.5

    cx = max(min(cx, iw - 1), 0)
    cy = max(min(cy, ih - 1), 0)

    l, t, r, b = int(cx), int(cy), math.ceil(cx), math.ceil(cy)
    dx, dy = cx - l, cy - t
    
    p1, p2, p3, p4 = matrix[t, l], matrix[t, r], matrix[b, l], matrix[b, r]
    f1, f2, f3, f4 = (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy
    value = f1 * p1 + f2 * p2 + f3 * p3 + f4 * p4
    return value


def ycbcr(b, g, r):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    return y, cb, cr


def bgr(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.3441 * (cb - 128) - 0.7139 * (cr - 128)
    b = y + 1.7718 * (cb - 128)
    return b, g, r


def highpass(gray, r):
    h, w = gray.shape
    cy, cx = int(h / 2), int(w / 2)
    
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    fshift[cy - r:cy + r, cx - r:cx + r] = 0
    iimg = np.fft.ifft2(np.fft.ifftshift(fshift))
    img = np.abs(iimg)
    return img


def lowpass(gray, r):
    h, w = gray.shape
    cy, cx = int(h / 2), int(w / 2)
    
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    pad = np.zeros_like(fshift)
    pad[cy - r:cy + r, cx - r:cx + r] = fshift[cy - r:cy + r, cx - r:cx + r]
    iimg = np.fft.ifft2(np.fft.ifftshift(fshift))
    img = np.abs(iimg)
    return img

