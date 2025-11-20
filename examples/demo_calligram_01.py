#!/usr/bin/env python3

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

from calligraph import (plut,
                        geom,
                        bspline,
                        bezier,
                        clipper,
                        dce,
                        svg,
                        config,
                        util,
                        fs,
                        diffvg_utils,
                        imaging,
                        spline_losses,
                        image_losses)

import time
device = config.device
dtype = torch.float32

def params():
    has_dropbox = os.path.isdir(os.path.expanduser('~/Dropbox/transfer_box/real'))
    if has_dropbox:
        output_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/outputs/'
    else:
        output_path = './generated/tests'

    save = 1

    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/bunny-sil.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/giraffe.jpg'
    text = 'GIRAFFE'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/sil-siggraph.jpg'
    text = 'SIGGRAPH'

    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/comp-siggraph.jpg'
    text = '/home/danielberio/Dropbox/transfer_box/data/calligraph/comp-siggraph.svg'


    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/sil-bunny-gpt.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/sil-yoga.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/comp-camel.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/bunny-sil.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/giraffe.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/umbrella.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/elephant.jpg'
    image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/SEAGULL.jpg'
    text = '.svg'
    #text = '/home/danielberio/Dropbox/transfer_box/data/calligraph/elephant.svg'
    # name = 'bunny1'
    # image_path = '/home/danielberio/Dropbox/transfer_box/data/calligraph/comp-%s.jpg'%name
    #text = '/home/danielberio/Dropbox/transfer_box/data/calligraph/comp-%s.svg'%name


    image_alpha = 1 #0.9
    
    stroke_w = 0.0 
    minw, maxw = 0.5, 2  # stroke width range
    degree =  5 #5 #5
    deriv = 2
    multiplicity = 1
    b_spline = True
    pspline = False
    ref_size_factor = 1.0

    lr_shape = 1.0 #1.5 
    num_opt_steps = 250 
    
    # OCR: no cls, av pool
    # BLIP: only cls
    use_ocr = True
    force_cls = None #False
    pool_no_cls = True

    if use_ocr:
        ocr_w = 400 
        if force_cls:
            ocr_w = 5000
    else:
        ocr_w = 10 

    
    ablation = ''

    smoothing_w = 13000 

    repulsion_subd = 10 
    repulsion_w = 10000 
    repulsion_d = 10 #7

    ds = 12 
    offset = 10 

    mse_w = 100 
    mse_mul = 1 

    seed = 1233
    suffix = ''

    save_every = 10
    return locals()

cfg = util.ConfigArgs(params())
output_path = cfg.output_path
if not cfg.save:
    output_path = '___'

cfg.ablation = cfg.ablation.lower()
if 'r' in cfg.ablation:
    cfg.repulsion_w = 0.0
if 's' in cfg.ablation:
    cfg.smoothing_w = 0.0
if 'o' in cfg.ablation:
    cfg.ocr_w = 0.0
if 'b' in cfg.ablation:
    cfg.b_spline = 0
if cfg.ablation:
    cfg.suffix += '_abl-' + cfg.ablation

saver = util.SaveHelper(__file__, output_path, 
                        suffix=cfg.suffix,
                        cfg=cfg)


input_img = Image.open(cfg.image_path).convert('L') #.resize((sz, sz))
sz, _ = input_img.size
box = geom.make_rect(0, 0, sz, sz)


has_svg = os.path.isfile(cfg.text)
layout_file = os.path.splitext(cfg.image_path)[0] + '_layout.json'
# If the layout file exists and we did not explictly define an init svg
# Load layout
if os.path.isfile(layout_file) and not has_svg:
    data = fs.load_json(layout_file)
    cfg.text = data['text']
    transforms = [np.array(mat) for mat in data['transforms']]
    outlines = []
    for S, tsm in zip(data['glyphs'], transforms):
        S = geom.tsm(tsm, [np.array(P) for P in S])
        outlines += S
else:
    # Else asume we provided a layout in cfg.text
    outlines = svg.load_svg(cfg.text)
    outlines = geom.fix_shape_winding(outlines)

# Use the input (either layout or SVG) as a target image for OCR
rast = imaging.ShapeRasterizer(box, sz)
rast.fill_shape(outlines)
text_img = rast.Image(invert=True)

# Prepare target area to cover
def opening(S, amt):
    return clipper.offset(clipper.offset(S, -amt, join_type='round'), amt, join_type='round')

im = np.array(input_img)/255
ctrs = clipper.multi_union([imaging.find_contours(1-im)]*2)
if cfg.offset:
    area = ctrs 
    complement = clipper.difference(ctrs, opening(ctrs, cfg.offset)) 
else:
    area = ctrs
    complement = []

rast = imaging.ShapeRasterizer(box, sz)
rast.fill_shape(area)
input_img = rast.Image(invert=True)
mse_img = input_img
mse_img = np.array(mse_img)/255
mse_img = 1-(1-mse_img)*cfg.image_alpha
img = np.array(input_img)/255
h, w = img.shape

# Sample outline and use as keypoints for splines
outlines = [geom.uniform_sample(P, cfg.ds, closed=True) for P in outlines]

##############################################
# Settings
verbose = False
overlap = False
ref_size = w/cfg.ref_size_factor
offset_variance = [ref_size, ref_size]
closed = True

diffvg_utils.cfg.one_channel_is_alpha = False
np.random.seed(cfg.seed)

##############################################
# Target tensor
target_img = np.array(input_img)/255
target_tensor = torch.tensor(target_img, device=device, dtype=dtype).contiguous()
background_image = np.ones_like(target_img)

def add_multiplicity(Q, noise=0.0):
    Q = np.kron(Q, np.ones((cfg.multiplicity, 1)))
    return Q + np.random.uniform(-noise, noise, Q.shape)

##############################################
# Initialization paths
scene = diffvg_utils.Scene()

closed = True
paths = []
for P in outlines:
    if cfg.b_spline:
        path = diffvg_utils.SmoothingBSpline(P,
                degree=cfg.degree,
                stroke_width=(cfg.stroke_w, False),
                pspline=cfg.pspline,
                closed=closed)
    else:
        path = diffvg_utils.CardinalSpline(P,
                                           stroke_width=(cfg.stroke_w, False),
                                           closed=closed)
    paths.append(path)

scene.add_shapes(paths, stroke_color=([0.0, 0.0, 0.0], False) if cfg.stroke_w > 0 else None,
                     fill_color=([0.0], False), split_primitives=False)

params = [(scene.get_points(), cfg.lr_shape)]
opt = diffvg_utils.SceneOptimizer(
    scene, params=params,
    num_steps=cfg.num_opt_steps,
    lr_min_scale=0.5
)


# schedulers = [util.step_cosine_lr_scheduler(opt, 0.0, 0.5, cfg.num_opt_steps) for opt in optimizers]
# #schedulers = []

##############################################
# Losses


losses = util.MultiLoss(verbose=verbose)

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




opt.add_loss('repulsion',
            spline_losses.make_repulsion_loss(cfg.repulsion_subd, False,
                                                single=False,
                                                signed=True, dist=cfg.repulsion_d),
             cfg.repulsion_w,
             ('shapes',))

opt.add_loss(
    "bbox", spline_losses.make_bbox_loss(geom.rect(0, 0, w, h)), 1.0, inputs=("points",)
)

from transformers import BlipProcessor, BlipModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
if cfg.use_ocr:
    name = "microsoft/trocr-base-handwritten"
    name = "microsoft/trocr-large-handwritten"
    name = "microsoft/trocr-large-printed"
    #name = "microsoft/trocr-base-printed"
    model = VisionEncoderDecoderModel.from_pretrained(name).to(device)
    processor = TrOCRProcessor.from_pretrained(name)
    cls = True
else:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    cls = True

if cfg.force_cls is not None:
    cls = cfg.force_cls

ocr_loss = image_losses.VisionEncoderLoss(model, processor,
                                            cls=cls,
                                            pool_no_cls=cfg.pool_no_cls,
                                              metric='L1', #'cosine', #l2',
                                             rgb=False)


opt.add_loss('ocr',
               ocr_loss, cfg.ocr_w,
             ('im', 'text_img'))

##############################################
# Begin visualization and optimize
plut.set_theme()

fig = plt.figure(figsize=(8,7))
gs = GridSpec(2, 3, height_ratios=[1.0, 0.4])
#gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, :])

first_frame = None
time_count = 0

def frame(step):
    global first_frame, time_count

    perf_t = time.perf_counter()

    opt.zero_grad()

    background_image = torch.ones((h, w), device=device)

    # Rasterize
    with util.perf_timer('render', verbose=verbose):
        im = opt.render(background_image)[:,:,0]

    opt.step(
        im=im,
        input_img=mse_img,
        text_img=text_img,
        shapes=scene.shapes,
        points=scene.get_points(),
        mse_mul=cfg.mse_mul,
    )

    elapsed = time.perf_counter() - perf_t
    time_count += elapsed
    lrs = 'lr %.3f'%(opt.optimizers[0].param_groups[0]['lr'])

    # Viz
    im = im.detach().cpu().numpy()
    if first_frame is None:
        first_frame = im

    must_save = step%cfg.save_every == cfg.save_every-1 or cfg.num_opt_steps==1
    must_show = True
    if cfg.headless and not must_save:
        must_show = must_save

    if must_show:
        if saver.valid:
            plt.suptitle(saver.name)
        plt.subplot(gs[0,0])
        plt.title('Startup - time: %.3f'%(time_count))
        amt = 0.5
        plt.imshow((np.array(text_img)/255)*amt + mse_img*(1-amt) , cmap='gray')
        plut.setup(box=geom.make_rect(0, 0, w, h), axis=True)


        plt.subplot(gs[0,1])
        if losses.has_loss('sds'):
            plt.title('Step %d, t %d, %s' %(step, int(sds.t_saved), lrs))
        else:
            plt.title('Step %d, %s'%(step, lrs))
        #plt.title('Step %d'%step)
        plt.imshow(im*amt + mse_img*(1-amt), cmap='gray', vmin=0, vmax=1)

        z = 0

        plt.subplot(gs[0,2])
        with torch.no_grad():
            S = []
            for i, group in enumerate(scene.shape_groups):
                #clr = np.ones(3)*quantized_hard[i] #  group.fill_color.detach().cpu().numpy()
                clr = np.ones(3)*group.fill_color.detach().cpu().numpy()
                #clr = np.ones(3)*colors[i].detach().cpu().numpy()
                for path in group.shapes:
                    points = path.param('points')
                    Q = points.detach().cpu().numpy()
                    X = path.samples(len(Q)*10).detach().cpu().numpy()
                    S.append(X)
        
            plut.fill(S, 'k')
            plut.stroke(S, 'k', lw=cfg.stroke_w*0.5, closed=True)
            if cfg.offset > 0:
                plut.fill(complement, 'k')

        plut.setup(box=geom.make_rect(0, 0, w, h))

        plt.subplot(gs[1,:])
        plt.title('Loss')
        opt.plot(50)
        plt.legend()

    if saver.valid and must_save:
        saver.clear_collected_paths()
        scene.save_json(saver.with_ext('.json'), complement=complement, area=area, ctrs=ctrs)
        cfg.save_yaml(saver.with_ext('.yaml'))
        plut.figure_image().save(saver.with_ext('.png'))
        saver.log_image('output', plt.gcf())
        saver.copy_file()


if cfg.headless:
    filename = ''
else:
    filename = saver.with_ext('.mp4')
plut.show_animation(fig, frame, cfg.num_opt_steps, filename=filename, headless=cfg.headless)
saver.finish()
