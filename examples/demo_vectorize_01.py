'''
Neural Image Abstraction Using Long Smoothing B-Splines
DEMO:
Quantized color vectorization
'''

from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image, ImageFilter
from easydict import EasyDict as edict
import torch
from skimage import feature
import torch.nn.functional as F
import os

from calligraph import (
    plut,
    geom,
    bspline,
    bezier,
    clipper,
    dce,
    sd,
    config,
    util,
    fs,
    ase_palette,
    diffvg_utils,
    tsp_art,
    segmentation,
    imaging,
    stroke_init,
    spline_losses,
    image_losses,
)

import time

device = config.device
dtype = torch.float32

# Color conversion utils
def rgb2lab(im):
    print(im.shape)
    im = to_batch(im)
    im = rgb_to_lab(im).permute((2, 3, 1, 0))
    res = im.squeeze(0).squeeze(-1)
    print("2lab", res.shape)
    return res


def lab2rgb(im):
    print(im.shape)
    im = to_batch(im)
    im = lab_to_rgb(im).permute((2, 3, 1, 0))
    res = im.squeeze(0).squeeze(-1)
    print("2rgb", res.shape)
    return res


def to_batch(im):
    im = im.unsqueeze(0)
    im = im.unsqueeze(-1)
    im = im.permute((3, 2, 0, 1))
    return im


def to_palette(im, n):
    # Naive palette from image
    im = im.quantize(n, method=Image.Quantize.MEDIANCUT, kmeans=n).convert("RGB")
    colors = im.getcolors()
    return [np.array(c[1]) / 255 for c in colors]


# Config
def params():
    output_path = "./outputs"
    save = True
    filename = "./data/bach.jpg"

    degree = 5  # Spline degree
    deriv = 3  # Smoothing degrees
    multiplicity = 1  # Keypoint multiplicity
    b_spline = 1
    pspline = 0  # If 1 (true) use discrete penalized spline_points

    num_voronoi_points = 50
    ds = 5
    lr_shape = 2.0
    lr_min_scale = 0.7
    lr_color = 1e-2
    num_opt_steps = 300

    use_color = 0

    clip_semantic_w = 0.0
    clip_model = "CLIPAG"
    clip_layer_weights = [(2, 1.0), (3, 1.0), (6, 1.0)]
    clipasso = True
    clip_w = 300.0
    semantic_w = 0.0

    sds = not clipasso
    sds_w = 1.0
    t_min, t_max = 0.5, 0.98

    cond_scale = 0.7  
    guess_mode = True
    ip_scale = 0.9  
    grad_method = "sds"
    grad_method = "ism"

    smoothing_w = 5.0 #0.5

    lab = False
    K = 3
    chans = 3
    if chans != 3:
        lab = False

    palette_im = './data/palettes/camo11.jpg'
    num_colors = 7
    gumbel_hard = 0
    tau_start = 1.0

    style_w = 10.0
    style_path = ""
    stroke_w = 0.0
    stroke_darkness = 0.5

    repulsion_subd = 10
    repulsion_w = 5000
    repulsion_d = 10

    rand_init = 2

    mse_w = 0.0
    mse_mul = 1  # Factor multiplying each mse blur level (> 1 emph low freq)

    seed = 1233
    suffix = ""

    save_every = 10
    return locals()


cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = "___"
saver = util.SaveHelper(
    __file__,
    output_path,
    use_wandb=False,  # cfg.headless,
    dropbox_folder=cfg.dropbox_folder,
    suffix=cfg.suffix,
    cfg=cfg,
)

palette = to_palette(Image.open(cfg.palette_im), cfg.num_colors)
palette = torch.tensor(palette, device=device, dtype=torch.float32)
K = len(palette)
cfg.gumbel_scale = 0.15  # 15 #0.15

if cfg.lab:
    palette = rgb2lab(palette)

# filename = './data/utah.jpg'
input_img = Image.open(cfg.filename).resize((400, 400))  # 512, 512))
saver.log_image("Input image", input_img)

img = np.mean(np.array(input_img) / 255, axis=-1)

style_im = cfg.palette_im

if cfg.style_path:  # is not None:
    style_im = cfg.style_path
style_img = Image.open(style_im).convert("RGB")

cond_img = feature.canny(img, 1.0)
cond_img = Image.fromarray((cond_img * 255).astype(np.uint8)).convert("RGB")
saver.log_image("Cond image", cond_img)

h, w = img.shape
box = geom.make_rect(0, 0, w, h)

##############################################
# Settings
verbose = False

overlap = False
closed = True

diffvg_utils.cfg.one_channel_is_alpha = False
np.random.seed(cfg.seed)

##############################################
# Target tensor
target_img = np.array(input_img) / 255
target_tensor = torch.tensor(target_img, device=device, dtype=dtype).contiguous()
background_image = np.ones_like(target_img)


def add_multiplicity(Q, noise=0.0):
    Q = np.kron(Q, np.ones((cfg.multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)


##############################################
# Initialization paths

# Saliency
from calligraph import ood_saliency

sal = segmentation.ood_saliency(input_img.convert("RGB"))[0] / 255
sal *= segmentation.clip_saliency(input_img)
density_map = ((sal - sal.min()) / (sal.max() - sal.min())) ** 2  # (sal*(1-img))

# Voronoi regions
points, verts = tsp_art.weighted_voronoi_sampling(
    density_map, cfg.num_voronoi_points, get_regions=True, nb_iter=50
)
verts = [geom.uniform_sample(V, cfg.ds * 2, closed=True) for V in verts]
verts = [V + np.random.uniform(-1, 1, V.shape) * cfg.rand_init for V in verts]

# Sort by saliency
rast = imaging.ShapeRasterizer(box, w)
im = np.array(img) / 255

colors = []
saliencies = []

for V in verts:
    rast.clear()
    rast.fill_shape(V)
    mask = np.array(rast.image()).astype(bool)
    s = np.mean(density_map[mask])
    v = np.mean(img[mask], axis=0)
    saliencies.append(s)
    colors.append(v)

# Sort by increasing salieny
I = np.argsort(saliencies)
colors = [colors[i] for i in I]
startup_paths = [verts[i] for i in I]
startup_paths = [add_multiplicity(P) for P in startup_paths]

scene = diffvg_utils.Scene()

closed = True

for P, clr in zip(startup_paths, colors):
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(
            P,
            degree=cfg.degree,
            stroke_width=(cfg.stroke_w, False),
            pspline=cfg.pspline,
            closed=closed,
        )
    else:
        path = diffvg_utils.CardinalSpline(
            P, stroke_width=(cfg.stroke_w, False), closed=closed
        )
    scene.add_shapes(
        [path],
        stroke_color=([0.0, 0.0, 0.0], False) if cfg.stroke_w > 0 else None,
        fill_color=([clr, clr, clr], True),
        split_primitives=False,
    )

# Create logits for palette
num_colors = len(startup_paths)
num_colors += 1
color_logits = torch.randn((num_colors, K), device=device) * 0.5
color_logits.requires_grad = True

params = [(scene.get_points(), cfg.lr_shape), ([color_logits], cfg.lr_color)]
opt = diffvg_utils.SceneOptimizer(
    scene, params=params, num_steps=cfg.num_opt_steps, lr_min_scale=cfg.lr_min_scale
)


##############################################
# Losses
opt.add_loss(
    "mse",
    image_losses.MultiscaleMSELoss(rgb=False),
    cfg.mse_w,
    inputs=("im", "input_img", "mse_mul"),
)
if cfg.b_spline:
    opt.add_loss(
        "deriv",
        spline_losses.make_deriv_loss(cfg.deriv, w),
        cfg.smoothing_w,
        inputs=("shapes",),
    )


opt.add_loss(
    "repulsion",
    spline_losses.make_repulsion_loss(
        cfg.repulsion_subd, False, single=True, signed=True, dist=cfg.repulsion_d
    ),
    cfg.repulsion_w,
    ("shapes",),
)

opt.add_loss(
    "bbox", spline_losses.make_bbox_loss(geom.rect(0, 0, w, h)), 1.0, inputs=("points",)
)


style_loss = image_losses.CLIPPatchLoss(
    rgb=False,
    image_prompts=[style_img],
    min_size=200,  # 28,
    n_cuts=64,
    distortion_scale=0.3,
    blur_sigma=0,
    model="CLIPAG",
    use_negative=False,
)  # clipag=cfg.use_clipag)

opt.add_loss("style", style_loss, cfg.style_w, inputs=("im",))


if cfg.clipasso:
    clip_loss = image_losses.CLIPVisualLoss(
        rgb=cfg.use_color,
        clip_model=cfg.clip_model,
        distortion_scale=0.5,
        crop_scale=(0.8, 0.9),  # 0.6, 0.7), # 0.9,
        layer_weights=cfg.clip_layer_weights,
        semantic_w=cfg.clip_semantic_w,
    )
    opt.add_loss("clip", clip_loss, cfg.clip_w, ("im", "input_img"))
if cfg.sds:
    sds = sd.SDSLoss(
        "A painting",
        rgb=False,
        controlnet="lllyasviel/sd-controlnet-canny",
        seed=cfg.seed,  # 777, #999,
        t_range=[cfg.t_min, cfg.t_max],
        guidance_scale=7.5,
        conditioning_scale=cfg.cond_scale,
        ip_adapter="ip-adapter-plus_sd15.bin",
        ip_adapter_scale=cfg.ip_scale,
        time_schedule="ism",
        grad_method=cfg.grad_method,
        guess_mode=False,
    )

    def sds_loss(im, step):
        return sds(
            im,
            cond_img,
            step,
            cfg.num_opt_steps,
            ip_adapter_image=input_img,
            grad_scale=1,
        )

    opt.add_loss("sds", sds_loss, cfg.sds_w, ("input_img", "step"))

##############################################
# Begin visualization and optimize

plut.set_theme()

fig = plt.figure(figsize=(8, 7))
gs = GridSpec(3, 3, height_ratios=[1.0, 0.25, 0.25])
gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :])

first_frame = None
time_count = 0


def frame(step):
    global first_frame, time_count

    perf_t = time.perf_counter()

    opt.zero_grad()

    # Setup colors
    tau_start = cfg.tau_start  # 5.0 #1.0
    tau_end = 0.1
    decay_rate = np.log(tau_end / tau_start) / cfg.num_opt_steps

    tau = max(tau_end, tau_start * np.exp(decay_rate * step))

    if cfg.gumbel_hard:
        soft_assign = F.gumbel_softmax(color_logits, tau=tau, hard=True)
    else:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(color_logits)))
        soft_assign = F.softmax(
            (color_logits + cfg.gumbel_scale * gumbel_noise) / tau, dim=-1
        )

    colors = soft_assign @ palette

    if cfg.gumbel_hard:
        quantized_hard = colors
        indices = soft_assign
    else:
        with torch.no_grad():
            indices = torch.argmax(color_logits, dim=-1)
            quantized_hard = palette[indices]

    if cfg.lab:
        colors = lab2rgb(colors)
        quantized_hard = lab2rgb(quantized_hard)

    quantized_hard = quantized_hard.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()

    for i, (group, color) in enumerate(zip(scene.shape_groups, colors[:-1])):
        # color = color_logits[i,:1]
        group.fill_color = color  # torch.tensor([1.0]).to(device)
        scene.groups[i]._fill_opt = color
        if cfg.stroke_w > 0.0:
            group.stroke_color = (
                color * cfg.stroke_darkness
            )  # torch.tensor([1.0]).to(device)
            scene.groups[i]._stroke_opt = color * cfg.stroke_darkness

    background_image = torch.ones((h, w, 3), device=device) * colors[-1]

    # Rasterize
    try:
        with util.perf_timer("render", verbose=verbose):
            im = opt.render(background_image)[:, :, :3].to(device)
    except RuntimeError as e:
        print(e)
        raise RuntimeError("error in render")

    mean_usage = soft_assign.mean(dim=0)
    uniform = torch.full_like(mean_usage, 1.0 / K)
    palette_usage_loss = 250 * F.mse_loss(mean_usage, uniform)
    # print('palette_usage_loss', palette_usage_loss)
    # print('Mean sum', mean_usage.sum().item())
    loss = palette_usage_loss

    opt.step(
        loss=loss,
        im=im,
        input_img=input_img,
        shapes=scene.shapes,
        points=scene.get_points(),
        mse_mul=cfg.mse_mul,
        step=step,
    )

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed
    lrs = "lr %.3f" % (opt.optimizers[0].param_groups[0]["lr"])

    # Viz
    im = im.detach().cpu().numpy()
    if first_frame is None:
        first_frame = im

    if saver.valid:
        plt.suptitle(saver.name)
    plt.subplot(gs[0, 0])
    plt.title("Startup - time: %.3f" % (time_count))
    amt = 0.5
    plt.imshow(first_frame * amt + target_img * (1 - amt), cmap="gray")

    # for group in scene.shape_groups:
    #     for shape in group.shapes:
    #         X = shape.samples(shape.num_points()*4).detach().cpu().numpy()
    #         plut.fill(X, np.ones(3)*group.fill_color.detach().cpu().numpy())
    # plut.setup()
    #     Q = path.param('points').detach().cpu().numpy()
    #     P = Q[::multiplicity]
    #     plut.stroke(P, 'r', lw=1.3, label='Keypoints' if i==0 else '', alpha=0.5)
    #     plut.stroke(Q, 'c', lw=0.5, label='Control Polygon' if i==0 else '', alpha=0.5)
    plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)

    plt.subplot(gs[0, 1])
    if opt.has_loss("sds"):
        plt.title("Step %d, t %d, %s" % (step, int(sds.t_saved), lrs))
    else:
        plt.title("Step %d, tau %.2f lr %s" % (step, tau, lrs))
    # plt.title('Step %d'%step)
    plt.imshow((im * 255).astype(np.uint8))  # , cmap='gray', vmin=0, vmax=1)

    plt.subplot(gs[0, 2])
    bg = np.ones((h, w, 3)) * quantized_hard[-1]
    plt.imshow(bg, vmin=0, vmax=1)

    z = 0

    # Render shapes with quantized colors
    with torch.no_grad():
        for i, group in enumerate(scene.shape_groups):
            clr = np.ones(3) * quantized_hard[i]  #
            stroke_clr = clr * cfg.stroke_darkness

            for path in group.shapes:
                points = path.param("points")
                Q = points.detach().cpu().numpy()
                X = path.samples(len(Q) * 10).detach().cpu().numpy()
                if closed:  # and not overlap:
                    fill = [
                        0.0,
                        0.0,
                        0.0,
                        0.1,
                    ]  # scene.shape_groups[i].fill_color.detach().cpu().numpy()
                    a = fill[-1]
                plut.fill(X[:, :2], clr[:3], zorder=z)
                if cfg.stroke_w > 0:
                    plut.stroke(
                        X[:, :2], stroke_clr, lw=cfg.stroke_w * 0.5, zorder=z + 1
                    )
                z += 3

    plut.setup(box=geom.make_rect(0, 0, w, h))

    plt.subplot(gs_sub[0])
    plt.title('Saliency')
    plt.imshow(density_map)
    plt.title('Style')
    plt.subplot(gs_sub[1])
    plt.imshow(style_img)
    plt.subplot(gs_sub[2])
    plt.title('Palette')
    for i, clr in enumerate(palette):
        plut.fill_rect(geom.make_rect(i, 0, 1, 1), clr.detach().cpu().numpy())
    plut.setup()
    plt.subplot(gs[2, :])
    plt.title("Loss")
    opt.plot(50)

    if saver.valid:
        
        if step % cfg.save_every == cfg.save_every - 1:
            saver.clear_collected_paths()
            scene.save_json(
                saver.with_ext(".json"),
                background_color=quantized_hard[-1],
                colors=quantized_hard[:-1],
                bg_index=indices[-1],
                indices=indices[:-1],
            )
            cfg.save_yaml(saver.with_ext(".yaml"))
            plut.figure_image().save(saver.with_ext(".png"))
            saver.log_image("output", plt.gcf())
            saver.copy_file()


plut.show_animation(
    fig,
    frame,
    cfg.num_opt_steps,
    filename=saver.with_ext(".mp4"),
    headless=cfg.headless,
)
saver.finish()
