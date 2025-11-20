#!/usr/bin/env python3
"""
)   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

image manipulation utils
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2

from . import geom


def brightness_contrast(img, brightness, contrast):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    return ImageEnhance.Contrast(img).enhance(contrast)


def expand_to_square(pil_img, background_color, get_padding=False):
    width, height = pil_img.size
    if width == height:
        result = pil_img
        pad = 0
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        pad = (width - height) // 2
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        pad = (height - width) // 2
    if get_padding:
        return result, pad
    return result


def pdf_from_image(img):
    """Transform an image into a probability distribution"""
    img = img / (np.sum(img) + 1e-10)
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    spline = RectBivariateSpline(x, y, img.T)

    def pdf(p):
        # No better way to do this?
        return np.array([spline(x, y, grid=False) for x, y in p])

    return pdf


def image_samples(img, n):
    """Rejection sampling on intensity treated as a pdf"""
    pdf = pdf_from_image(img)
    h, w = img.shape
    grids_x, grids_y = np.meshgrid(
        np.linspace(0, w - 1, w),
        np.linspace(0, h - 1, h),
    )

    grids = np.vstack([np.ravel(grids_x), np.ravel(grids_y)]).T
    max_pdf = np.max(pdf(grids))

    samples = []
    while len(samples) < n:
        x_rand = np.random.uniform(0, w, n)
        y_rand = np.random.uniform(0, h, n)
        candidates = np.vstack([x_rand, y_rand]).T
        u = np.random.uniform(0, max_pdf, n)
        pdf_vals = pdf(candidates)
        accepted = candidates[u < pdf_vals]
        samples.append(accepted)

    return np.vstack(samples)


def kernel(size=None):
    if size is None:
        size = 5
    return np.ones((size, size), np.uint8)


def morpho_erode(im, size=None, iterations=1):
    return cv2.erode(im, kernel(size), iterations=iterations)


def morpho_dilate(im, size=None, iterations=1):
    return cv2.dilate(im, kernel(size), iterations=iterations)


def morpho_open(im, size=None, iterations=1):
    return morpho_dilate(
        morpho_erode(im, size=size, iterations=iterations),
        size=size,
        iterations=iterations,
    )


def morpho_close(im, size=None, iterations=1):
    return morpho_erode(
        morpho_dilate(im, size=size, iterations=iterations),
        size=size,
        iterations=iterations,
    )


def morpho_pass(im, size=None, iterations=1):
    return im


def find_contours(im, invert=False, thresh=127, eps=0.0):
    """Utility function to get contours compatible with py5canvas.
    Assumes a grayscale image as a result
    The eps parameter controls the amount of simplification (if > 0)
    """
    if im.dtype == np.uint8:
        _, thresh_img = cv2.threshold(im, thresh, 256, int(invert))
    elif im.max() == 1:
        if invert:
            im = 1 - im
        thresh_img = (im > 0.5).astype(np.uint8)
    else:
        thresh_img = cv2.convertScaleAbs(im.astype(float))
        if invert:
            thresh_img = 1 - thresh_img

    contours, hierarchy = cv2.findContours(
        thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    S = []
    for ctr in contours:
        if eps > 0:
            ctr = cv2.approxPolyDP(ctr, eps, True)
        if len(ctr) > 3:
            S.append(np.vstack([ctr[:, 0, 0], ctr[:, 0, 1]]).T)
    return S


def shape_to_outline(S):
    s = ImageDraw.Outline()
    for P in S:
        s.move(*P[0])
        for p in P[1:]:
            s.line(*p)
        s.close()
    return s


class ShapeRasterizer:
    """Helper class to rasterize shapes via PIL"""

    def __init__(self, src_rect, raster_size=512, debug_draw=False):
        if type(raster_size) not in [list, tuple, np.ndarray]:
            raster_size = (raster_size, raster_size)
        elif type(raster_size) == np.ndarray:
            raster_size = raster_size.shape

        dst_rect = geom.make_rect(0, 0, *raster_size)  # raster_size, raster_size)
        if src_rect is None:
            src_rect = dst_rect
        self.box = dst_rect  # geom.scale_rect(dst_rect, 1)  # 1.2)

        self.mat = geom.rect_in_rect_transform(src_rect, dst_rect)
        self.inv_mat = np.linalg.inv(self.mat)
        self.scale = np.sqrt(np.linalg.det(self.mat))
        self.raster_size = raster_size
        self.debug_draw = debug_draw
        self.context = self.create_context()

    def clear(self, color=0):
        self.set_context(self.create_context(color))

    def create_context(self, color=0):
        """Create a new image with given size"""
        im = Image.new("L", self.raster_size, color)
        draw = ImageDraw.Draw(im)
        self.context = (im, draw)
        return im, draw

    def set_context(self, context):
        if context is not None:
            self.context = context

    def fill_circle(self, p, r, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        xy = geom.affine_transform(self.mat, p)
        r = r * self.scale
        draw.ellipse([xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r], fill=color)

    def fill_circles(self, centers, radii, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        for p, r in zip(centers, radii):
            self.fill_circle(p, r, color, context)

    def fill_shape(self, S, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        if type(S) != list:
            S = [S]
        S = geom.affine_transform(self.mat, S)
        draw.shape(shape_to_outline(S), color)

    def line(self, a, b, color=255, lw=1, context=None):
        self.set_context(context)
        im, draw = self.context
        a = geom.affine_transform(self.mat, a)
        b = geom.affine_transform(self.mat, b)
        draw.line((tuple(a), tuple(b)), fill=255, width=lw)  # , joint='curve')

    def stroke_shape(self, S, lw=1, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        if type(S) != list:
            S = [S]
        S = geom.affine_transform(self.mat, S)
        for P in S:
            for a, b in zip(P, P[1:]):
                draw.line((tuple(a), tuple(b)), fill=255, width=lw, joint="curve")

    def fill_polygon(self, P, color=255, context=None):
        self.set_context(context)
        im, draw = self.context
        P = geom.affine_transform(self.mat, P)
        P = [(float(p[0]), float(p[1])) for p in P]
        draw.polygon(P, fill=color)  # , outline=color)

    def blit(self, context_src, context=None):
        self.set_context(context)
        im, draw = self.context
        draw.bitmap((0, 0), context_src[0], 255)

    def contours(self, context=None):
        self.set_context(context)
        im, draw = self.context
        ctrs = find_contours(np.array(im))
        if not ctrs:
            return ctrs
        ctrs = geom.affine_transform(self.inv_mat, ctrs)
        return ctrs

    def image(self, context=None):
        self.set_context(context)
        im, draw = self.context
        return im

    def get_image(self, context=None):
        self.set_context(context)
        im, draw = self.context
        return np.array(im)

    def Image(self, invert=False, context=None):
        im = self.get_image(context)
        if invert:
            im = 255 - im
        return Image.fromarray(im).convert("L")
