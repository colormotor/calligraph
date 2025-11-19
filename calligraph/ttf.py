#!/usr/bin/env python3
"""
 )   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

TTF loading utils
"""

import freetype as ft
from . import bezier, fs
import os
import numpy as np


class FontDatabase:
    def __init__(self, path):
        paths = fs.files_in_dir(path, "ttf")
        self.db = {os.path.splitext(os.path.basename(f))[0]: f for f in paths}

    def shapes(self, font, txt, **kwargs):
        return font_string_to_shapes(self.db[font], txt, **kwargs)


def font_string_to_beziers(font, txt, size=30, spacing=1.0, merge=True):
    """Load a font and convert the outlines for a given string to cubic bezier curves,
    if merge is True, simply return a list of all bezier curves,
    otherwise return a list of lists with the bezier curves for each glyph"""

    face = ft.Face(font)
    face.set_char_size(64 * size)
    slot = face.glyph

    x = 0
    beziers = []
    previous = 0
    for c in txt:
        face.load_char(c, ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP)
        bez = glyph_to_cubics(face, x)

        if merge:
            beziers += bez
        else:
            beziers.append(bez)

        kerning = face.get_kerning(previous, c)
        # print(slot.advance.x)
        x += (
            slot.advance.x + kerning.x
        ) * spacing  # Unsure if kerning needs to be multiplied here
        previous = c

    return beziers


def font_string_to_shapes(font, txt, merge=True, subd=40, **kwargs):
    beziers = font_string_to_beziers(font, txt, merge=merge, **kwargs)
    if merge:
        return [bezier.bezier_piecewise(Cp, subd) for Cp in beziers]
    else:
        return [[bezier.bezier_piecewise(Cp, subd) for Cp in bez] for bez in beziers]


def glyph_to_cubics(face, x=0):
    """Convert current font face glyph to cubic beziers"""

    def linear_to_cubic(Q):
        a, b = Q
        return [a + (b - a) * t for t in np.linspace(0, 1, 4)]

    def quadratic_to_cubic(Q):
        return [
            Q[0],
            Q[0] + (2 / 3) * (Q[1] - Q[0]),
            Q[2] + (2 / 3) * (Q[1] - Q[2]),
            Q[2],
        ]

    beziers = []
    pt = lambda p: np.array([p.x + x, -p.y])  # Flipping here since freetype has y-up
    last = lambda: beziers[-1][-1]

    def move_to(a, beziers):
        beziers.append([pt(a)])

    def line_to(a, beziers):
        Q = linear_to_cubic([last(), pt(a)])
        beziers[-1] += Q[1:]

    def conic_to(a, b, beziers):
        Q = quadratic_to_cubic([last(), pt(a), pt(b)])
        beziers[-1] += Q[1:]

    def cubic_to(a, b, c, beziers):
        beziers[-1] += [pt(a), pt(b), pt(c)]

    face.glyph.outline.decompose(
        beziers, move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to
    )
    beziers = [np.array(C).astype(float) for C in beziers]
    return beziers
