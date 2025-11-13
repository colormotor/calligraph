"""
)   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/

Basic SVG loading utils
"""

from importlib import reload
import svgpathtools as svg
import xml.etree.ElementTree as ET
from functools import reduce
import numpy as np
from . import bezier
from . import geom
import pdb
import re
import matplotlib.colors as mcolors


def to_pt(c):
    """convert complex number to np vector"""
    return np.array([c.real, c.imag])


# Elliptic arc code borrowed from https://github.com/dbrnz/flatmap-ol-maker
def clamp(value, min_value, max_value):
    # ======================================
    return min(max(value, min_value), max_value)


def svg_angle(u, v):
    # ===================
    dot = u.real * v.real + u.imag * v.imag
    length = np.sqrt(u.real**2 + u.imag**2) * np.sqrt(v.real**2 + v.imag**2)
    angle = np.arccos(clamp(dot / length, -1, 1))
    if (u.real * v.imag - u.imag * v.real) < 0:
        angle = -angle
    return angle


def elliptic_arc_point(c, r, phi, eta):
    # ======================================
    return complex(
        c.real
        + r.real * np.cos(phi) * np.cos(eta)
        - r.imag * np.sin(phi) * np.sin(eta),
        c.imag
        + r.real * np.sin(phi) * np.cos(eta)
        + r.imag * np.cos(phi) * np.sin(eta),
    )


def elliptic_arc_derivative(r, phi, eta):
    # ========================================
    return complex(
        -r.real * np.cos(phi) * np.sin(eta) - r.imag * np.sin(phi) * np.cos(eta),
        -r.real * np.sin(phi) * np.sin(eta) + r.imag * np.cos(phi) * np.cos(eta),
    )


def cubic_bezier_control_points(c, r, phi, eta1, eta2):
    # ======================================================
    alpha = (
        np.sin(eta2 - eta1)
        * (np.sqrt(4 + 3 * np.power(np.tan((eta2 - eta1) / 2), 2)) - 1)
        / 3
    )
    P1 = elliptic_arc_point(c, r, phi, eta1)
    d1 = elliptic_arc_derivative(r, phi, eta1)
    Q1 = complex(P1.real + alpha * d1.real, P1.imag + alpha * d1.imag)
    P2 = elliptic_arc_point(c, r, phi, eta2)
    d2 = elliptic_arc_derivative(r, phi, eta2)
    Q2 = complex(P2.real - alpha * d2.real, P2.imag - alpha * d2.imag)
    return (P1, Q1, Q2, P2)


def cubic_beziers_from_arc(arc):  # r, phi, flagA, flagS, p1, p2):
    # ========================================================
    r = arc.radius
    p1 = arc.start
    p2 = arc.end
    phi = geom.radians(arc.rotation)
    flagA = False  # arc.large_arc
    flagS = True  # arc.sweep
    # irint(arc)
    r_abs = complex(abs(r.real), abs(r.imag))
    d = complex((p1.real - p2.real), (p1.imag - p2.imag))
    p = complex(
        np.cos(phi) * d.real / 2 + np.sin(phi) * d.imag / 2,
        -np.sin(phi) * d.real / 2 + np.cos(phi) * d.imag / 2,
    )
    p_sq = complex(p.real**2, p.imag**2)
    r_sq = complex(r_abs.real**2, r_abs.imag**2)

    ratio = p_sq.real / r_sq.real + p_sq.imag / r_sq.imag
    if ratio > 1:
        scale = np.sqrt(ratio)
        r_abs = complex(scale * r_abs.real, scale * r_abs.imag)
        r_sq = complex(r_abs.real**2, r_abs.imag**2)

    dq = r_sq.real * p_sq.imag + r_sq.imag * p_sq.real
    pq = (r_sq.real * r_sq.imag - dq) / dq
    q = np.sqrt(max(0, pq))
    if flagA == flagS:
        q = -q

    cp = complex(
        q * r_abs.real * p.imag / r_abs.imag, -q * r_abs.imag * p.real / r_abs.real
    )
    c = complex(
        cp.real * np.cos(phi) - cp.imag * np.sin(phi) + (p1.real + p2.real) / 2.0,
        cp.real * np.sin(phi) + cp.imag * np.cos(phi) + (p1.imag + p2.imag) / 2.0,
    )

    lambda1 = svg_angle(
        complex(1, 0),
        complex((p.real - cp.real) / r_abs.real, (p.imag - cp.imag) / r_abs.imag),
    )
    delta = svg_angle(
        complex((p.real - cp.real) / r_abs.real, (p.imag - cp.imag) / r_abs.imag),
        complex((-p.real - cp.real) / r_abs.real, (-p.imag - cp.imag) / r_abs.imag),
    )
    delta = delta - 2 * np.pi * np.floor(delta / (2 * np.pi))
    if not flagS:
        delta -= 2 * np.pi
    lambda2 = lambda1 + delta

    t = lambda1
    dt = np.pi / 4
    curves = []
    while (t + dt) < lambda2:
        control_points = (
            cp for cp in cubic_bezier_control_points(c, r_abs, phi, t, t + dt)
        )
        curves.append(svg.CubicBezier(*control_points))
        t += dt
    control_points = (
        cp for cp in cubic_bezier_control_points(c, r_abs, phi, t, lambda2)
    )
    curves.append(svg.CubicBezier(*(tuple(control_points)[:3]), p2))
    return curves


def to_bezier(piece):
    """convert a line or Bezier segment to control points"""
    one3d = 1.0 / 3
    if type(piece) == svg.path.Line:
        a, b = to_pt(piece.start), to_pt(piece.end)
        return [[a, a + (b - a) * one3d, b + (a - b) * one3d, b]]
    elif type(piece) == svg.path.CubicBezier:
        return [
            [
                to_pt(piece.start),
                to_pt(piece.control1),
                to_pt(piece.control2),
                to_pt(piece.end),
            ]
        ]
    elif type(piece) == svg.path.QuadraticBezier:
        QP0 = to_pt(piece.start)
        QP1 = to_pt(piece.control)
        QP2 = to_pt(piece.end)
        CP1 = QP0 + 2 / 3 * (QP1 - QP0)
        CP2 = QP2 + 2 / 3 * (QP1 - QP2)
        return [[QP0, CP1, CP2, QP2]]
    elif type(piece) == svg.path.Arc:
        bezs = sum([to_bezier(ap) for ap in cubic_beziers_from_arc(piece)], [])
        print("Arc bezs")
        # print(bezs)
        return bezs

    raise ValueError


def path_to_bezier(path):
    """convert SVG path to a Bezier control points"""
    pieces = sum([to_bezier(piece) for piece in path], [])
    bezier = [pieces[0][0]] + sum([piece[1:] for piece in pieces], [])
    return np.vstack(bezier)


def to_segment(piece):
    """convert a line or Bezier segment to control points"""
    return [to_pt(piece.start), to_pt(piece.end)]


def path_to_polyline(path):
    """convert SVG path to a Bezier control points"""
    pieces = [to_segment(piece) for piece in path]
    try:
        bezier = [pieces[0][0]] + sum([piece[1:] for piece in pieces], [])
    except IndexError as e:
        print("Corrupt segment in path")
    return np.vstack(bezier)


def split_compound_paths(paths, data=[]):
    """Split compound paths, since svgpathtools does not do that by default"""
    import re

    split_paths = []
    if data:
        split_data = []
    for i, path in enumerate(paths):
        if not path:
            continue
        s = path.d()
        # split at occurrences of moveto commands
        sub_d = filter(None, re.split("[Mm]", s))
        # indices (without moveto) of splits
        lens = [len(list(filter(None, re.split("[A-z]", d)))) - 1 for d in sub_d]
        # cum sum
        split_inds = [0] + reduce(lambda c, x: c + [c[-1] + x], lens, [0])[1:]
        chunk = [svg.Path(*path[a:b]) for a, b in zip(split_inds, split_inds[1:])]
        if data:
            split_data += [data[i]] * len(chunk)
        split_paths += chunk
    if data:
        return split_paths, split_data
    return split_paths


def hex_to_rgb(hex_color):
    """Convert hex color (#RRGGBB) to an RGB array in range [0,1]"""
    try:
        return np.array(mcolors.hex2color(hex_color))  # Normalize 0-1
    except ValueError:
        return None  # Invalid hex


def rgb_string_to_array(rgb_str):
    """Convert 'rgb(r,g,b)' string to normalized [0,1] RGB array"""
    try:
        values = re.findall(r"\d+", rgb_str)  # Extract numbers
        if len(values) == 3:  # Ensure it's an RGB triplet
            return np.array(
                [int(values[0]) / 255, int(values[1]) / 255, int(values[2]) / 255]
            )
    except Exception as e:
        print(e)
        pass
    return None


def parse_color(color_str):
    """Convert color string (hex, rgb, or named) to a 3D numpy array in range [0,1]"""
    if not color_str or color_str.lower() == "none":
        print(color_str)
        return None  # No color

    color_str = color_str.strip().lower()

    if color_str.startswith("#"):  # Hex format
        return hex_to_rgb(color_str)
    elif color_str.startswith("rgb"):  # RGB format
        return rgb_string_to_array(color_str)
    elif color_str in mcolors.CSS4_COLORS:  # Named colors (e.g., "red")
        return hex_to_rgb(mcolors.CSS4_COLORS[color_str])
    print("unrecognized", color_str)
    return None  # Unrecognized color


#
def extract_path_colors(svg_path):
    """Extract colors (stroke and fill) from an SVG file"""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}  # Namespace handling

    colors = []
    for elem in root.findall(".//svg:path", ns):
        stroke = parse_color(elem.attrib.get("stroke", "none"))
        fill = parse_color(elem.attrib.get("fill", "none"))
        colors.append({"stroke": stroke, "fill": fill})

    return colors


def svg_to_beziers(path, get_colors=False):
    """Load Bezier curves from a SVG file"""

    doc = svg.Document(path)
    paths = doc.paths()
    if get_colors:
        colors = extract_path_colors(path)  # Get colors
        paths, colors = split_compound_paths(paths, colors)
    else:
        paths = split_compound_paths(paths)
        colors = []
    # paths, attributes = svg.svg2paths(path)

    beziers = [path_to_bezier(path) for path in paths]
    if colors:
        return beziers, colors
    return beziers


def load_svg(file_path, subd=40, get_colors=False):
    if get_colors:
        paths, colors = svg_to_beziers(file_path, get_colors=True)
    else:
        paths = svg_to_beziers(file_path)
    S = [bezier.bezier_piecewise(path, subd) for path in paths]
    if get_colors:
        return S, colors
    return S


def load_svg_polylines(path):
    """Load Bezier curves from a SVG file"""
    # paths, attributes = svg.svg2paths(path)
    doc = svg.Document(path)
    print(doc)
    paths = doc.paths()
    print("Num paths", len(paths))
    paths = split_compound_paths(paths)
    print("Num split paths", len(paths))
    polylines = [path_to_polyline(path) for path in paths if len(path) > 0]
    return polylines


def load_svg_bezier_chains(file_path, subd=40):
    paths = svg_to_beziers(file_path)
    return [path for path in paths]


def save_svg_polylines(
    S, file_path, stroke_widths=None, closed=False, dimensions=None, unit="mm", **kwargs
):
    if type(closed) == bool:
        closed = [closed for _ in S]
    valid = [len(P) > 1 for P in S]
    print(valid)
    S = [geom.close(P) if c else P for P, c in zip(S, closed)]
    if stroke_widths is None:
        stroke_widths = [1.0] * len(S)
    S = [P for i, P in enumerate(S) if valid[i]]
    stroke_widths = [w for i, w in enumerate(stroke_widths) if valid[i]]
    if dimensions is not None:
        dimensions = (
            "%.2f%s" % (dimensions[0], unit),
            "%.2f%s" % (dimensions[1], unit),
        )
    paths = [
        svg.Path(
            *[svg.Line(start=complex(*a), end=complex(*b)) for a, b in zip(P, P[1:])]
        )
        for P in S
    ]  # if len(P) > 1]
    svg.wsvg(
        paths, filename=file_path, stroke_widths=stroke_widths, dimensions=dimensions
    )  # **kwargs)


