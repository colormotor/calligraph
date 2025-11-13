'''
)   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/


Matplotlib helpers
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Path, PathPatch
from matplotlib.colors import ListedColormap
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.path import Path

import matplotlib
import numpy as np
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter
from io import BytesIO
from PIL import Image
from matplotlib.font_manager import FontProperties
import os, time
from . import geom

def show_animation(fig, frame, num_frames, ax=None, filename='', fps=30, headless=False):
    import traceback

    frames = []
    ani = None

    if ax is None:
        ax = plt.gca()

    state = lambda: None
    state.playing = True

    def _frame(i):

        plt.clf()
        try:
            frame(i)
        except Exception as e:
            if ani is not None:
                ani.event_source.stop()
            print(e)
            raise e

        fig = plt.gcf()
        if filename:
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            frames.append(np.array(image)[:,:,:3])

        if i == num_frames - 1:
            print("Animation ended")
            if ani is not None:
                ani.event_source.stop()
            state.playing = False
            # time.sleep(0.5)
            # if norepeat:
            #     #raise StopIteration

            #     plt.close(fig)
            #     raise AttributeError
            #     #plt.close()


    norepeat = os.getenv("QUIT_ANIMATION") == "1"

    if not headless:
        ani = FuncAnimation(fig, _frame, frames=num_frames,
                        blit=False,
                        repeat=not norepeat,
                        interval=1000/fps)

        # Ugly workaround because func anim does not let us close
        if norepeat:
            import threading
            def monitor_animation():
                while state.playing:
                    time.sleep(0.5)
                print("Closing animation")
                plt.close(fig)
            monitor_thread = threading.Thread(target=monitor_animation)
            monitor_thread.start()
        try:
            plt.show()
        except AttributeError as e:
            print("Closing")
    else:
        for i in range(num_frames):
            print('Headless: step %d of %d'%(i+1, num_frames))
            _frame(i)

    if filename and frames:
        print('saving %d frames'%len(frames))
        import cv2
        fmt = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, _ = frames[0].shape
        video_writer = cv2.VideoWriter(filename, fmt, fps, (w, h))
        for frame in frames:
            video_writer.write(frame[:,:,::-1])
        video_writer.release()
        print("Finished saving")

def figure_pixels(w, h):
    plt.figure(figsize=(w/100, h/100), dpi=100)
    
def figure_image(close=False, adjust=True):
    if adjust:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    if close:
        plt.close(plt.gcf())
    image = Image.open(buf)
    return image

def text_to_image(letter, image_size=(100, 100), font_size=60, font_path=None, grayscale=True):
    # Thanks ChatGPT
    if font_path:
        font_prop = FontProperties(fname=font_path, size=font_size)
    else:
        font_prop = FontProperties(size=font_size)  # Default font

    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
    ax.text(0.5, 0.4, letter, ha='center', va='center', fontproperties=font_prop)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)


    image = Image.open(buf)
    if grayscale:
        image = image.convert("L")
    return image

def font_to_image(text, image_size, padding=0, font_path=None, grayscale=True,
                  return_outline=False,
                  separate_glyphs=False,
                  vspace=1.0,
                  subd=20):
    # Load font
    font_size = 60
    if font_path:
        font_prop = FontProperties(fname=font_path, size=font_size)
    else:
        font_prop = FontProperties(size=font_size)

    lines = text.splitlines()

    # Get vector path
    vertices = []
    codes = []

    for i, text in enumerate(lines):
        text_path = TextPath((0, 0), text, prop=font_prop)
        tvertices = text_path.vertices.copy()
        tvertices[:, 1] *= -1
        tcodes = text_path.codes

        box = geom.bounding_box(tvertices)
        center = geom.rect_center(box)
        tvertices = geom.tsm(geom.trans_2d(-center + [0, font_size*i]), tvertices)
        vertices.append(tvertices)
        codes.append(tcodes)
    vertices = np.vstack(vertices)
    codes = np.concatenate(codes)
    # for text in lines:
    #     text_path = TextPath((0, 0), text, prop=font_prop)
    #     vertices = text_path.vertices.copy()
    #     codes = text_path.codes


    # Transform to image space
    box = geom.make_rect(0, 0, image_size[0], image_size[1])
    src_box = geom.bounding_box(vertices)
    mat = geom.rect_in_rect_transform(src_box, box, padding=padding)
    vertices = geom.tsm(mat, vertices)
    text_path = Path(vertices, codes)

    # Draw to image
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
    #ax = fig.add_axes([0, 0, 1, 1])
    patch = PathPatch(text_path, facecolor="k", edgecolor="none")
    ax.add_patch(patch)

    setup(box=box)
    # Key for sizing!
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf)
    if grayscale:
        image = image.convert("L")

    if not return_outline:
        return image

    # Optional: Return outline samples too
    curves = []
    i = 0
    while i < len(codes):
        code = codes[i]
        if code == Path.MOVETO:
            start = vertices[i]
            i += 1
            curves.append([start])
        elif code == Path.LINETO:
            end = vertices[i]
            line = np.linspace(start, end, subd)
            curves[-1].append(line[1:])
            start = end
            i += 1
        elif code == Path.CURVE3:  # Quadratic Bézier
            ctrl = vertices[i]
            end = vertices[i + 1]
            t = np.linspace(0, 1, subd).reshape(-1, 1)
            pts = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * end
            curves[-1].append(pts[1:])
            start = end
            i += 2
        elif code == Path.CURVE4:  # Cubic Bézier
            ctrl1 = vertices[i]
            ctrl2 = vertices[i + 1]
            end = vertices[i + 2]
            t = np.linspace(0, 1, subd).reshape(-1, 1)
            pts = (
                (1 - t) ** 3 * start +
                3 * (1 - t) ** 2 * t * ctrl1 +
                3 * (1 - t) * t ** 2 * ctrl2 +
                t ** 3 * end
            )
            curves[-1].append(pts[1:])
            start = end
            i += 3
        elif code == Path.CLOSEPOLY:
            i += 1
        else:
            i += 1

    return image, [np.vstack(X) for X in curves]

def font_to_outlines(text, rect=None, padding=0, font_path=None,
                     font_size=60, grayscale=True, return_outline=False,
                     vspace=1.0, subd=20, kerning=1.0):
    # Load font
    if font_path:
        font_prop = FontProperties(fname=font_path, size=font_size)
    else:
        font_prop = FontProperties(size=font_size)

    lines = text.splitlines()

    all_vertices = []
    all_codes = []
    all_glyphs = []
    y_offset = 0

    for line in lines:
        x_cursor = 0
        for char in line:
            if char.isspace():
                # Approximate space width
                space_width = font_size * 0.3
                x_cursor += space_width*kerning
                continue

            char_path = TextPath((0, 0), char, prop=font_prop)
            bounds = char_path.get_extents()
            vertices = char_path.vertices.copy()
            vertices[:, 1] *= -1  # Flip y axis

            vertices[:, 0] += x_cursor
            vertices[:, 1] += y_offset

            all_glyphs.append((vertices.copy(), char_path.codes.copy()))

            advance = bounds.width*kerning + bounds.x0
            x_cursor += advance

        y_offset += font_size * vspace

    # Combine all vertices and codes into a single path
    vertices = np.vstack([v for v, c in all_glyphs])
    codes = np.concatenate([c for v, c in all_glyphs])

    mat = np.eye(3)
    if rect is not None:
        mat = geom.rect_in_rect_transform(geom.bounding_box(vertices),
                                          rect, padding=padding)
        vertices = geom.tsm(mat, vertices)

    # Now generate curves per glyph
    outlines = []
    for glyph_vertices, glyph_codes in all_glyphs:
        # Apply the same transformation
        glyph_vertices = geom.tsm(mat, glyph_vertices)

        curves = []
        i = 0
        while i < len(glyph_codes):
            code = glyph_codes[i]
            if code == Path.MOVETO:
                start = glyph_vertices[i]
                i += 1

                curves.append([[start]])
            elif code == Path.LINETO:
                end = glyph_vertices[i]
                line = np.linspace(start, end, subd)
                curves[-1].append(line[1:])
                start = end
                i += 1
            elif code == Path.CURVE3:  # Quadratic Bézier
                ctrl = glyph_vertices[i]
                end = glyph_vertices[i + 1]
                t = np.linspace(0, 1, subd).reshape(-1, 1)
                pts = (1 - t) ** 2 * start + 2 * (1 - t) * t * ctrl + t ** 2 * end
                curves[-1].append(pts[1:])
                start = end
                i += 2
            elif code == Path.CURVE4:  # Cubic Bézier
                ctrl1 = glyph_vertices[i]
                ctrl2 = glyph_vertices[i + 1]
                end = glyph_vertices[i + 2]
                t = np.linspace(0, 1, subd).reshape(-1, 1)
                pts = (
                    (1 - t) ** 3 * start +
                    3 * (1 - t) ** 2 * t * ctrl1 +
                    3 * (1 - t) * t ** 2 * ctrl2 +
                    t ** 3 * end
                )
                curves[-1].append(pts[1:])
                start = end
                i += 3
            elif code == Path.CLOSEPOLY:
                i += 1
            else:
                i += 1

        outlines.append([np.vstack(segs) for segs in curves])

    return outlines


# def font_to_image(text, image_size, padding=10, font_path=None, grayscale=True):

#     font_size = 60
#     if font_path:
#         font_prop = FontProperties(fname=font_path, size=font_size)
#     else:
#         font_prop = FontProperties(size=font_size)  # Default font

#     text_path = TextPath((0, 0), text, prop=font_prop)

#     vertices = text_path.vertices.copy()  # Points along the path
#     codes = text_path.codes        # Commands (e.g., MOVETO, LINETO)

#     vertices[:,1] *= -1 # Y is flipped
#     box = geom.make_rect(0, 0, image_size[0], image_size[1])
#     src_box = geom.bounding_box(vertices)
#     mat = geom.rect_in_rect_transform(src_box, box, padding=padding)
#     vertices = geom.tsm(mat, vertices)
#     text_path = Path(vertices, codes)

#     fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)
#     patch = PathPatch(text_path, facecolor="k", edgecolor="none")
#     ax.add_patch(patch)
#     setup(box=box)

#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close(fig)
#     buf.seek(0)

#     image = Image.open(buf)
#     if grayscale:
#         image = image.convert("L")
#     return image

def set_theme():
    plt.style.use('fivethirtyeight') #fivethirtyeight')
    import matplotlib as mpl
    label_size = 7
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.labelsize'] = 10 #u'large' # 12 #'medium'
    mpl.rcParams['font.size']  = 10
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['lines.linewidth'] = 2.5
    mpl.rcParams["figure.facecolor"] = 'ffffff'
    #mpl.rcParams["axes.facecolor"] = 'cccccc'
    mpl.rcParams["savefig.facecolor"] = 'ffffff'
    plt.rcParams['lines.solid_capstyle'] = 'round'

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def inset_image(img, **kwargs):
    ax = plt.gca()
    arg = {'width':'25%',
           'height':'25%',
           'loc':'upper left',
           'borderpad':1}
    arg.update(kwargs)
    inset_ax = inset_axes(ax, **arg) #width="30%", height="30%", loc='upper left')
    inset_ax.imshow(img)
    #inset_ax.axis('off')

    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('grey')   # Border color
        spine.set_linewidth(1)       # Border thickness

    inset_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def stroke(S, clr='k', closed=False, **kwargs):
    if type(S)==list and not S:
        # print('Empty shape')
        return
    if geom.is_compound(S):
        for P in S:
            stroke(P, clr=clr, closed=closed, **kwargs)
        return

    # Send out
    P = [p for p in S]
    if closed:
        P = P + [P[0]]

    P = np.array(P).T

    if len(P.shape) < 2:
        return

    plt.plot(P[0], P[1], color=mpl.colors.to_rgb(clr), **kwargs)


def fill(S, clr, **kwargs):
    if type(S)==list and not S:
        # print('Empty shape')
        return

    if not geom.is_compound(S):
        S = [S]

    if not S:
        # print('Empty shape')
        return

    path = []
    cmds = []
    for P in S:
        if not len(P):
            continue
        path += [p for p in P] + [P[0]]
        cmds += [Path.MOVETO] + [Path.LINETO for p in P[:-1]] + [Path.CLOSEPOLY]
    if path:
        plt.gca().add_patch(PathPatch(Path(path, cmds), color=clr, fill=True, linewidth=0, **kwargs))


def fill_stroke(S, clr, strokeclr, **kwargs):
    if not S:
        # print('Empty shape')
        return
    if not geom.is_compound(S):
        S = [S]

    path = []
    cmds = []
    for P in S:
        path += [p for p in P] + [P[0]]
        cmds += [Path.MOVETO] + [Path.LINETO for p in P[:-1]] + [Path.CLOSEPOLY]
    plt.gca().add_patch(PathPatch(Path(path, cmds), facecolor=clr, edgecolor=strokeclr, fill=True,  **kwargs))


def stroke_rect(rect, clr='k', plot=True, **kwargs):
    x, y = rect[0]
    w, h = rect[1] - rect[0]

    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=False, edgecolor=clr, **kwargs))

def fill_rect(rect, clr, **kwargs):
    x, y = rect[0]
    w, h = rect[1] - rect[0]
    plt.gca().add_patch(
        patches.Rectangle((x, y), w, h, fill=True, facecolor=clr, **kwargs))

def fill_circle(pos, radius, clr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=True, facecolor=clr, **kwargs)) #alpha=alpha, zorder=zorder))

def stroke_circle(pos, radius, clr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=False, edgecolor=clr, **kwargs)) #alpha=alpha, zorder=zorder))

def fill_stroke_circle(pos, radius, clr, strokeclr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=True, facecolor=clr, edgecolor=strokeclr, **kwargs)) #alpha=alpha, zorder=zorder))

def stroke_circle(pos, radius, clr, **kwargs):
    plt.gca().add_patch(
        patches.Circle(pos, radius, fill=False, edgecolor=clr, **kwargs)) #alpha=alpha, zorder=zorder))

def plot_gauss(mu, Sigma, w=1., color='c', alpha=0.5, scale=1., fill=True, centersize=2):
    v, w = np.linalg.eigh(Sigma) # <- ordered with smalles first
    v = 2. * np.sqrt(v) * scale
    u = w[1] / np.linalg.norm(w[1])
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = patches.Ellipse(mu, v[0]*2, v[1]*2, angle=angle, color=color, alpha=alpha, fill=fill, linewidth=1)
    plt.gca().add_patch(ell)
    fill_circle(mu, centersize, color)

def draw_markers(P, color, marker='o', **kwargs):
    P = np.array(P)
    if type(color) == str:
        plt.plot(P[:,0], P[:,1], color, linestyle='None', marker=marker, **kwargs)
    else:
        plt.plot(P[:,0], P[:,1], color=color, linestyle='None', marker=marker, **kwargs)

def draw_line(a, b, clr, **kwargs):
    p = np.vstack([a,b])
    plt.plot(p[:,0], p[:,1], color=clr, solid_capstyle='round', dash_capstyle='round', **kwargs)

def det22(mat):
    return mat[0,0] * mat[1,1] - mat[0,1]*mat[1,0]

def draw_arrow(a, b, clr, alpha=1., head_width=0.15, head_length=None, overhang=0.3, zorder=None, **kwargs):
    if head_length is None:
        head_length = head_width

    linewidth = 1.0
    if 'lw' in kwargs:
        linewidth = kwargs['lw']
    if 'linewidth' in kwargs:
        linewidth = kwargs['linewidth']
    head_width /= linewidth
    head_length /= linewidth
    # Uglyness, still does not work
    axis = plt.gca()
    trans = axis.transData.inverted()
    scale = np.sqrt(det22(trans.get_matrix()))*axis.figure.dpi*100
    head_width = (linewidth*head_width)*scale
    head_length = (linewidth*head_length)*scale
    a, b  = np.array(a), np.array(b)
    d = b - a

    draw_line(a, b - geom.normalize(d)*head_length*0.5, clr, linewidth=linewidth)
    plt.arrow(a[0], a[1], d[0], d[1], lw=0.5, overhang=overhang,
              head_width=head_width, head_length=head_length, length_includes_head=True,
              fc=clr, ec='none', zorder=zorder)

def set_axis_limits(box, pad=0, invert=True, ax=None, y_limits_only=False):
    # UNUSED
    if ax is None:
        ax = plt.gca()

    xlim = [box[0][0]-pad, box[1][0]+pad]
    ylim = [box[0][1]-pad, box[1][1]+pad]

    ax.set_ylim(ylim)
    ax.set_ybound(ylim)
    if not y_limits_only:
        ax.set_xlim(xlim)
        ax.set_xbound(xlim)

    # Hack to get matplotlib to actually respect limits?
    stroke_rect([geom.vec(xlim[0], ylim[0]), geom.vec(xlim[1], ylim[1])], 'r', plot=False, alpha=0)
    # ax.set_clip_on(True)
    if invert:
        ax.invert_yaxis()

def set_axis_limits(P, pad=0, invert=True, ax=None, y_limits_only=False):
    if ax is None:
        ax = plt.gca()

    if type(P) == tuple or (type(P)==list and len(P)==2):
        box = P
        xlim = [box[0][0]-pad, box[1][0]+pad]
        ylim = [box[0][1]-pad, box[1][1]+pad]
    else:
        if type(P) == list:
            P = np.hstack(P)
        xlim = [np.min(P[0,:])-pad, np.max(P[0,:])+pad]
        ylim = [np.min(P[1,:])-pad, np.max(P[1,:])+pad]
    ax.set_ylim(ylim)
    ax.set_ybound(ylim)
    if not y_limits_only:
        ax.set_xlim(xlim)
        ax.set_xbound(xlim)

    # Hack to get matplotlib to actually respect limits?
    stroke_rect([geom.vec(xlim[0],ylim[0]), geom.vec(xlim[1], ylim[1])], 'r', alpha=0, plot=False)
    # ax.set_clip_on(True)
    if invert:
        ax.invert_yaxis()

def figure_inches(w, h, dpi=None):
    return figure((w,h))

figure = plt.figure

def show(title='', padding=0, box=None, axis=False, invert_y=True, file='', debug_box=False):
    if title:
        plt.title(title)

    setup(invert_y, axis, box, debug_box)

    if file:
        plt.savefig(file, transparent=False)

    plt.show()

def setup(invert_y=True, axis=False, box=None, debug_box=False):
    ax = plt.gca()
    ax.axis('scaled')
    if not axis:
        ax.axis('off')
    else:
        ax.axis('on')
    if invert_y:
        ax.invert_yaxis()
    if debug_box and box is not None:
        stroke_rect(box, 'r', plot=False)
    if box is not None:
        set_axis_limits(box, invert=invert_y, ax=ax, y_limits_only=False)

categorical_palettes = {
    'Tabular':[
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
(1.0, 0.4980392156862745, 0.054901960784313725),
(1.0, 0.7333333333333333, 0.47058823529411764),
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(1.0, 0.596078431372549, 0.5882352941176471),
(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
(0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
(0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
    ],
'Dark2_8':[
(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
(0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
(0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
(0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
(0.4, 0.6509803921568628, 0.11764705882352941),
(0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
(0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
(0.4, 0.4, 0.4)
],
'Custom':[
[104.0/255, 175.0/255, 252.0/255],
[66.0/255, 47.0/255, 174.0/255],
[71.0/255, 240.0/255, 163.0/255],
[29.0/255, 104.0/255, 110.0/255],
[52.0/255, 218.0/255, 234.0/255],
[45.0/255, 93.0/255, 168.0/255],
[219.0/255, 119.0/255, 230.0/255],
[165.0/255, 46.0/255, 120.0/255],
[171.0/255, 213.0/255, 51.0/255],
[29.0/255, 109.0/255, 31.0/255],
[143.0/255, 199.0/255, 137.0/255],
[226.0/255, 50.0/255, 9.0/255],
[93.0/255, 242.0/255, 62.0/255],
[94.0/255, 64.0/255, 40.0/255],
[247.0/255, 147.0/255, 2.0/255],
[255.0/255, 0.0/255, 135.0/255],
[226.0/255, 150.0/255, 163.0/255],
[216.0/255, 197.0/255, 152.0/255],
[97.0/255, 8.0/255, 232.0/255],
[243.0/255, 212.0/255, 38.0/255]
],
'Paired_12':[ # Okeish
(0.6509803921568628, 0.807843137254902, 0.8901960784313725),
(0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
(0.6980392156862745, 0.8745098039215686, 0.5411764705882353),
(0.2, 0.6274509803921569, 0.17254901960784313),
(0.984313725490196, 0.6039215686274509, 0.6),
(0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
(0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
(1.0, 0.4980392156862745, 0.0),
(0.792156862745098, 0.6980392156862745, 0.8392156862745098),
(0.41568627450980394, 0.23921568627450981, 0.6039215686274509),
(1.0, 1.0, 0.6),
(0.6941176470588235, 0.34901960784313724, 0.1568627450980392)
],
'Tableau_20':[
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
(1.0, 0.4980392156862745, 0.054901960784313725),
(1.0, 0.7333333333333333, 0.47058823529411764),
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(1.0, 0.596078431372549, 0.5882352941176471),
(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
(0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
(0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
],
'Bold_10':[
(0.4980392156862745, 0.23529411764705882, 0.5529411764705883),
(0.06666666666666667, 0.6470588235294118, 0.4745098039215686),
(0.2235294117647059, 0.4117647058823529, 0.6745098039215687),
(0.9490196078431372, 0.7176470588235294, 0.00392156862745098),
(0.9058823529411765, 0.24705882352941178, 0.4549019607843137),
(0.5019607843137255, 0.7294117647058823, 0.35294117647058826),
(0.9019607843137255, 0.5137254901960784, 0.06274509803921569),
(0.0, 0.5254901960784314, 0.5843137254901961),
(0.8117647058823529, 0.10980392156862745, 0.5647058823529412),
(0.9764705882352941, 0.4823529411764706, 0.4470588235294118)
],
    'Prism_10':[
(0.37254901960784315, 0.27450980392156865, 0.5647058823529412),
(0.11372549019607843, 0.4117647058823529, 0.5882352941176471),
(0.2196078431372549, 0.6509803921568628, 0.6470588235294118),
(0.058823529411764705, 0.5215686274509804, 0.32941176470588235),
(0.45098039215686275, 0.6862745098039216, 0.2823529411764706),
(0.9294117647058824, 0.6784313725490196, 0.03137254901960784),
(0.8823529411764706, 0.48627450980392156, 0.0196078431372549),
(0.8, 0.3137254901960784, 0.24313725490196078),
(0.5803921568627451, 0.20392156862745098, 0.43137254901960786),
(0.43529411764705883, 0.25098039215686274, 0.4392156862745098)
    ],
'ColorBlind_10':[
(0.0, 0.4196078431372549, 0.6431372549019608),
(1.0, 0.5019607843137255, 0.054901960784313725),
(0.6705882352941176, 0.6705882352941176, 0.6705882352941176),
(0.34901960784313724, 0.34901960784313724, 0.34901960784313724),
(0.37254901960784315, 0.6196078431372549, 0.8196078431372549),
(0.7843137254901961, 0.3215686274509804, 0.0),
(0.5372549019607843, 0.5372549019607843, 0.5372549019607843),
(0.6352941176470588, 0.7843137254901961, 0.9254901960784314),
(1.0, 0.7372549019607844, 0.4745098039215686),
(0.8117647058823529, 0.8117647058823529, 0.8117647058823529)
],
    'BlueRed_12':[
(0.17254901960784313, 0.4117647058823529, 0.6901960784313725),
(0.7098039215686275, 0.7843137254901961, 0.8862745098039215),
(0.9411764705882353, 0.15294117647058825, 0.12549019607843137),
(1.0, 0.7137254901960784, 0.6901960784313725),
(0.6745098039215687, 0.3803921568627451, 0.23529411764705882),
(0.9137254901960784, 0.7647058823529411, 0.6078431372549019),
(0.4196078431372549, 0.6392156862745098, 0.8392156862745098),
(0.7098039215686275, 0.8745098039215686, 0.9921568627450981),
(0.6745098039215687, 0.5294117647058824, 0.38823529411764707),
(0.8666666666666667, 0.788235294117647, 0.7058823529411765),
(0.7411764705882353, 0.0392156862745098, 0.21176470588235294),
(0.9568627450980393, 0.45098039215686275, 0.47843137254901963)
    ],
    'plut_categorical_12':[
(1.0, 0.42745098039215684, 0.6823529411764706),
(0.8313725490196079, 0.792156862745098, 0.22745098039215686),
(0.0, 0.7450980392156863, 1.0),
(0.9215686274509803, 0.6745098039215687, 0.9803921568627451),
(0.6196078431372549, 0.6196078431372549, 0.6196078431372549),
(0.403921568627451, 0.8823529411764706, 0.7098039215686275),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(0.6039215686274509, 0.8156862745098039, 1.0),
(0.8862745098039215, 0.5215686274509804, 0.26666666666666666),
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(1.0, 0.4980392156862745, 0.054901960784313725),
(0.36470588235294116, 0.6941176470588235, 0.35294117647058826)
    ],
'cyan2':[
    (0.9, 0.9, 1.0),
    (0.2, 0.8, 1.0),
    ]
}


default_palette_name = 'plut_categorical_12' #
default_palette_name = None # 'Tabular' #'Tabular' #'BlueRed_12' #'Tabular' #'BlueRed_12' #'Tabular' #'BlueRed_12'# 'Paired_12' #OK #'ColorBlind_10' #OK # 'Dark2_8'# OKeish 'Bold_10' #OK #
def get_default_colors():
    if default_palette_name is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return colors

    return categorical_palettes[default_palette_name]
    #return plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    #return plt.rcParams['axes.prop_cycle'].by_key()['color'] + list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + list(plt.get_cmap('tab20c').colors)

def categorical_palette(name=None):
    if type(name)==list:
        return name
    if name is None:
        name = default_palette_name
    return categorical_palettes[name]

def categorical(i, palette=None, rep=1):
    clrs = categorical_palette(palette)
    return mpl.colors.to_rgb(clrs[i%len(clrs)])

    return categorical_cmap(name, rep)

def categorical_cmap(name=None, rep=1):
    return ListedColormap(categorical_palette(name)*rep)

def default_color(i):
    clrs = get_default_colors() #plt.rcParams['axes.prop_cycle'].by_key()['color']
    return mpl.colors.to_rgb(clrs[i%len(clrs)])

def default_color_alpha(i, alpha):
    rgb = default_color(i) #default_colors[i%len(default_colors)]
    return list(rgb) + [alpha]

def cmap(v, colormap='turbo'): #'PuRd'):
    c = matplotlib.cm.get_cmap(colormap)
    return c(v)

def overlay_bar(ax, value):
    """Overlays a bar indicator on the right of an existing subplot without modifying its axes."""
    ax_position = ax.get_position()
    fig = ax.figure

    # Create an inset axis for the bar
    bar_ax = fig.add_axes([ax_position.x1 + 0.01, ax_position.y0, 0.02, ax_position.height])

    # Background bar (full height)
    bar_ax.bar(0, 1, color='lightgray', width=1, align='edge')

    # Foreground bar (starting from bottom, going up)
    bar_ax.bar(0, value, color='c', width=1, align='edge')

    # Add label near the bar
    bar_ax.text(0, 1.05, 'endt', fontsize=12, horizontalalignment='center')

    # Hide axes
    bar_ax.set_xlim(-0.5, 0.5)
    bar_ax.set_ylim(0, 1)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    bar_ax.spines['top'].set_visible(False)
    bar_ax.spines['bottom'].set_visible(False)
    bar_ax.spines['left'].set_visible(False)
    bar_ax.spines['right'].set_visible(False)

    return bar_ax

fig = figure
