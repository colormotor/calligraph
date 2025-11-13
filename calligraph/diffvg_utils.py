#!/usr/bin/env python3
"""Utilities wrapping diffvg for facilitating differentiable rendering of more
complex primitives Different shapes derive from the Shape class, and can be
added to a Scene object. Shape parameters can be set as a tuple, with first
parameter specifying the initial value (can be numpy or list) and second value a
boolean that specifies if the parameter has gradients

"""

from importlib import reload
import torch
import pydiffvg
from collections import defaultdict
import numpy as np
from . import config, bspline, bezier, geom, fs
import pdb
from easydict import EasyDict as edict

device = config.diffvg_device
pydiffvg.set_use_gpu(config.has_gpu)

cfg = lambda: None
cfg.one_channel_is_alpha = True

shape_classes = {}


def skip_kwargs(skip, kwargs):
    return {k: v for k, v in kwargs.items() if k not in skip}


def load_and_render(path, background, **kwargs):
    from PIL import Image

    scene = Scene()
    scene.load_json(path, **kwargs)
    im = scene.render(background).detach().cpu().numpy()
    return Image.fromarray((im * 255).astype(np.uint8))


def load_scene(path, **kwargs):
    scene = Scene()
    scene.load_json(path, **kwargs)
    return scene


def load_strokes(
    path, subd=20, get_scene=False, get_centerlines=False, clip_w=0.0, **kwargs
):
    scene = Scene()
    scene.load_json(path, **kwargs)
    strokes = []
    centerlines = []
    for group in scene.shape_groups:
        for shape in group.shapes:
            n = shape.num_points()
            X = shape.samples(n * subd).detach().cpu().numpy()
            if not shape.has_varying_width():
                w = shape.param("stroke_width").detach().cpu().numpy()
                X = np.hstack([X, np.ones((len(X), 1)) * w])
            centerlines.append(X[:, :2])
            if clip_w > 0:
                X[:, 2] = np.maximum(0, X[:, 2] - clip_w)
            S = geom.thick_curve(X)
            strokes.append(S)
    if get_centerlines:
        return strokes, centerlines
    if get_scene:
        return strokes, scene
    return strokes


def load_stroke_control_points(path, get_scene=False, **kwargs):
    scene = Scene()
    scene.load_json(path, **kwargs)
    strokes = []
    for group in scene.shape_groups:
        for shape in group.shapes:
            P = shape.param("points").detach().cpu().numpy()
            if shape.has_varying_width():
                stroke_w = shape.param("stroke_width").detach().cpu().numpy()
                P = np.hstack([P, stroke_w.reshape(-1, 1)])
            strokes.append(P)
    if get_scene:
        return strokes, scene
    return strokes


def sample_strokes(path, subd=20, get_scene=False, clip_w=0.0, **kwargs):
    scene = Scene()
    scene.load_json(path, **kwargs)
    strokes = []
    for group in scene.shape_groups:
        for shape in group.shapes:
            n = shape.num_points()
            X = shape.samples(n * subd).detach().cpu().numpy()
            if X.shape[1] > 2:
                if clip_w > 0:
                    X[:, 2] = np.maximum(0, X[:, 2] - clip_w)
            strokes.append(X)
    if get_scene:
        return strokes, scene
    return strokes


def load_areas(path, subd=20, get_scene=False, **kwargs):
    scene = Scene()
    scene.load_json(path, **kwargs)
    areas = []
    for group in scene.shape_groups:
        for shape in group.shapes:
            n = shape.num_points()
            X = shape.samples(n * subd).detach().cpu().numpy()
            areas.append(X[:, :2])
    if get_scene:
        return areas, scene
    return areas


class SceneOptimizer:
    """Optimization wrapper for differentiable rendering scenes.
    Handles optimization of scene parameters with configurable loss functions,
    learning rate scheduling, and gradient updates.
    """

    def __init__(
        self,
        scene,
        params,
        num_steps,
        anneal=True,
        schedule_start=0.0,
        lr_min_scale=0.2,
        verbose=False,
        opt=lambda params, lr: torch.optim.Adam(params, lr, betas=(0.9, 0.999)),
    ):
        self.scene = scene
        self.optimizers = []
        for p, lr in params:
            self.optimizers.append(opt(p, lr))

        if anneal:
            self.schedulers = [
                util.step_cosine_lr_scheduler(
                    opt, schedule_start, lr_min_scale, num_steps
                )
                for opt in self.optimizers
            ]
        else:
            self.schedulers = []

        self.num_steps = num_steps

        self.losses = util.MultiLoss(verbose)
        self.loss_inputs = {}
        self.verbose = verbose

    def add_loss(self, name, loss, weight, inputs=None):
        self.losses.add(name, loss, weight)
        self.loss_inputs[name] = inputs

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def render(self, background_image, num_samples=2, prefiltering=False):
        self.im = self.scene.render(
            background_image, num_samples=num_samples, prefiltering=prefiltering
        ).to(device)
        return self.im

    def loss(self, **kwargs):
        loss_args = {}

        for name, input_format in self.loss_inputs.items():
            if input_format is None:
                inputs = (self.im,)
            else:
                inputs = []
                for inp in input_format:
                    if inp is None:
                        inputs.append(self.im)
                    elif inp in kwargs:
                        inputs.append(kwargs[inp])
                    else:
                        print(kwargs)
                        breakpoint()
                        raise ValueError(inp, "invalid input to loss", name)
            loss_args[name] = inputs
        self._loss = self.losses(**loss_args)
        return self._loss

    def step(self, compute_loss=True, **kwargs):
        if compute_loss:
            self.loss(**kwargs)
        with util.perf_timer("backwards", verbose=self.verbose):
            self._loss.backward()
            for opt in self.optimizers:
                opt.step()
            for sched in self.schedulers:
                sched.step()

    def implot(self):
        from slimgui import implot

        if implot.begin_plot("Loss", size=[-1, 200]):
            implot.setup_axes(
                None, None, implot.AxisFlags.AUTO_FIT, implot.AxisFlags.AUTO_FIT
            )
            for i, (key, kloss) in enumerate(self.losses.losses.items()):
                if key == "total" or not self.losses.has_loss(key):
                    # There is a bug in 'total'
                    continue
                # implot.plot_line(f'{key}:{np.round(kloss[-1],2)}##trello', np.array(kloss))
                implot.plot_line(f"{key}", np.array(kloss[-50:]))
            implot.end_plot()


class Scene:
    """Abstraction of a DiffVG scene, handles colors, params and rendering"""

    scale = 1.0

    def __init__(self):
        self.groups = []
        self.primitives = []
        self.shapes = []
        self.params = defaultdict(list)
        self.shape_groups = []
        self.transforms = []

    def add_shapes(self, shapes, split_primitives=True, transform=None, **kwargs):
        params = args_to_params(**kwargs)
        _params = {
            "stroke_color": ([0.0, 0.0, 0.0, 1.0], False),
            "fill_color": (None, False),
            "opacity": (1.0, False),
        }
        _params.update(params)
        _params = {key: convert_param(p) for key, p in _params.items()}

        # Add stroke and fill color
        for key, p in _params.items():
            if p is not None:
                self.params[key].append(p)
        # Add parameters for each shape
        for shape in shapes:
            for key, p in shape.params.items():
                if p is not None:
                    self.params[key].append(p)

        primitives = sum([s.primitives for s in shapes], [])
        ind = len(self.primitives)

        # Hacky way to pass the transform for a single shape to its group
        # Using this for ellipses
        shape_to_canvas = None
        shape_to_canvas_proxy = None
        for shape in shapes:
            shape_to_canvas = shape.shape_to_canvas()
            if shape_to_canvas is not None:
                shape_to_canvas_proxy = shape
                break

        if transform is not None:  #
            # if shape_to_canvas_proxy is not None:
            #    raise ValueError('Shape provides transform already so it is currently incompatible with an external one')
            # shape_to_canvas_proxy = transform
            for shape in shapes:
                shape.transform = transform
            self.transforms.append(transform)

        diffvg_groups = []
        # NB we use 'get_color_param' here, this allows to optimize for different number of channels or grayscale
        if split_primitives:
            # Split primitives is used to draw
            if (
                _params["fill_color"] is not None
                and _params["stroke_color"] is not None
            ):
                group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor(list(range(ind, ind + len(primitives)))),
                    fill_color=get_color_param(
                        _params["fill_color"], _params["opacity"]
                    ),
                    use_even_odd_rule=False,
                    stroke_color=None,
                )
                group._fill_opt = _params["fill_color"]
                group._stroke_opt = _params["stroke_color"]
                group._opacity_opt = _params["opacity"]
                group._shape_to_canvas_proxy = shape_to_canvas_proxy
                self.groups.append(group)
                diffvg_groups.append(group)
                fill_clr = None
            else:
                fill_clr = get_color_param(_params["fill_color"], _params["opacity"])

            for i, prim in enumerate(primitives):
                group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor(list(range(ind + i, ind + i + 1))),
                    fill_color=fill_clr,
                    use_even_odd_rule=False,
                    stroke_color=get_color_param(
                        _params["stroke_color"], _params["opacity"]
                    ),
                )
                group._fill_opt = _params["fill_color"]
                group._stroke_opt = _params["stroke_color"]
                group._opacity_opt = _params["opacity"]
                group._shape_to_canvas_proxy = shape_to_canvas_proxy
                self.groups.append(group)
                diffvg_groups.append(group)
        else:
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor(list(range(ind, ind + len(primitives)))),
                use_even_odd_rule=False,
                fill_color=get_color_param(_params["fill_color"], _params["opacity"]),
                stroke_color=get_color_param(
                    _params["stroke_color"], _params["opacity"]
                ),
            )
            group._fill_opt = _params["fill_color"]
            group._stroke_opt = _params["stroke_color"]
            group._opacity_opt = _params["opacity"]
            group._shape_to_canvas_proxy = shape_to_canvas_proxy
            self.groups.append(group)
            diffvg_groups.append(group)
        self.shapes += shapes
        self.primitives += primitives
        self.shape_groups.append(
            edict(
                {
                    "shapes": shapes,
                    "diffvg_groups": diffvg_groups,
                    "fill_color": _params["fill_color"],
                    "stroke_color": _params["stroke_color"],
                }
            )
        )

        return self.shape_groups[-1]

    def transformed_points(self):
        pts = []
        for group in self.groups:
            mat = group._shape_to_canvas_proxy.transform
            for i in group.shape_ids:
                P = self.primitives[i].points
                P = torch.hstack(
                    [P, torch.ones((len(P), 1), device=device, dtype=torch.float32)]
                )
                pts.append((mat @ P.T).T[:, :2])
        return pts

    def add_shape(self, shape, **kwargs):
        self.add_shapes(
            [shape], split_primitives=isinstance(shape, StepwisePolygon), **kwargs
        )

    def render(
        self,
        background_image,
        postupdate=None,
        prefiltering=False,
        size=None,
        num_samples=2,
        seed=0,
    ):
        for group in self.groups:
            group.fill_color = get_color_param(group._fill_opt, group._opacity_opt)
            group.stroke_color = get_color_param(group._stroke_opt, group._opacity_opt)
            if group._shape_to_canvas_proxy is not None:
                group.shape_to_canvas = group._shape_to_canvas_proxy.shape_to_canvas()
        for tsm in self.transforms:
            tsm.shape_to_canvas()

        for shape in self.shapes:
            shape.update()

        if postupdate is not None:
            postupdate()

        if prefiltering:
            num_samples = 1

        if background_image is not None:
            background_image = torch.tensor(background_image, dtype=torch.float32).to(
                device
            )

            if len(background_image.shape) == 2:
                background_image = background_image[:, :, np.newaxis]
                background_image = background_image.repeat(1, 1, 3)
            h, w, _ = background_image.shape
        else:
            h, w = size

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            w, h, self.primitives, self.groups, use_prefiltering=prefiltering
        )
        img = pydiffvg.RenderFunction.apply(
            w, h, num_samples, num_samples, seed, None, *scene_args
        )
        # print('Rendered image with device', img.device)
        if background_image is not None:
            img = img[:, :, 3:4] * img[:, :, :3] + background_image * (
                1 - img[:, :, 3:4]
            )
            img = img[:, :, :3]
            return img
        else:
            return img

    def get_params(self, key, only_grad=True, numpy=False):
        params = []
        for p in self.params[key]:
            if p is not None:
                if only_grad and not p.requires_grad:
                    pass
                else:
                    params.append(p)
        if numpy:
            return [p.detach().cpu().numpy() for p in params]
        return params

    def get_points(self, only_grad=True, **kwargs):
        return self.get_params("points", only_grad, **kwargs)

    def get_stroke_widths(self, only_grad=True, **kwargs):
        return self.get_params("stroke_width", only_grad, **kwargs)

    def get_stroke_colors(self, only_grad=True, **kwargs):
        return self.get_params("stroke_color", only_grad, **kwargs)

    def get_fill_colors(self, only_grad=True, **kwargs):
        return self.get_params("fill_color", only_grad, **kwargs)

    def get_transform_positions(self, only_grad=True):
        return [
            tsm.pos
            for tsm in self.transforms
            if not only_grad or (only_grad and tsm.pos.requires_grad)
        ]

    def get_transform_rotations(self, only_grad=True):
        return [
            tsm.rot
            for tsm in self.transforms
            if not only_grad or (only_grad and tsm.rot.requires_grad)
        ]

    def get_transform_scales(self, only_grad=True):
        return [
            tsm.scale
            for tsm in self.transforms
            if not only_grad or (only_grad and tsm.scale.requires_grad)
        ]

    def to_dict(self, **kwargs):
        json = {"groups": []}
        for i, group in enumerate(self.shape_groups):
            g = {}
            g["shapes"] = [shape.to_dict() for shape in group.shapes]
            if group.fill_color is not None:
                g["fill_color"] = group.fill_color.detach().cpu().numpy()
            if group.stroke_color is not None:
                g["stroke_color"] = group.stroke_color.detach().cpu().numpy()
            json["groups"].append(g)
        if kwargs:
            json["extra"] = kwargs
        return json

    def save_json(self, path, **kwargs):
        json = self.to_dict(**kwargs)
        fs.save_json(json, path)

    def load_json(
        self,
        path,
        default_cls="Path",
        split_primitives=False,
        transform=None,
        subd=None,
    ):
        json = fs.load_json(path)

        if "groups" not in json:
            json = {"groups": [{"shapes": [json]}]}

        for g in json["groups"]:
            fill_color = g.get("fill_color", None)
            stroke_color = g.get("stroke_color", None)
            shapes = []
            for shape_rec in g["shapes"]:
                # if 'type' in shape_rec:
                #     classtype = shape_rec['type']
                # else:
                #     classtype = default_cls
                # try:
                #     Cls = eval(classtype)
                # except:
                #     Cls = shape_classes[classtype]
                # shape = Cls(**{key:val for key,val in shape_rec.items() if key != 'type'})
                shape = shape_from_dict(shape_rec, default_cls)
                if subd is not None:
                    nump = len(shape_rec["points"])
                    X = shape.samples(nump * subd).detach().cpu().numpy()
                    args = {}
                    if X.shape[1] > 2:
                        args["stroke_width"] = X[:, 2]
                    else:
                        if "stroke_width" in shape_rec:
                            args["stroke_width"] = shape_rec["stroke_width"]
                    if "closed" in shape_rec:
                        args["closed"] = shape_rec["closed"]
                    shape = Polygon(X, **args)
                shapes.append(shape)
            self.add_shapes(
                shapes,
                fill_color=fill_color,
                stroke_color=stroke_color,
                transform=transform,
                split_primitives=split_primitives,
            )
        if "extra" in json:
            self.extra = edict(json["extra"])
        else:
            self.extra = {}


def shape_from_dict(shape_rec, default_cls="Path"):
    if "type" in shape_rec:
        classtype = shape_rec["type"]
        print(classtype)
    else:
        if "spline_degree" in shape_rec:
            classtype = "DynamicBSpline"
        else:
            classtype = default_cls
        # print("Default class")
    try:
        Cls = eval(classtype)
    except:
        Cls = shape_classes[classtype]
    shape = Cls(**{key: val for key, val in shape_rec.items() if key != "type"})
    return shape


class Shape:
    """Generic shape"""

    def __init__(self):
        self.degree = 1
        self.transform = None
        pass

    def get_degree(self):
        """Hack so we can get the effective degree of a higher level primitive, e.g. bsplines"""
        return self.degree

    def length(self):
        P = self.param("points")
        D = torch.diff(P, axis=0)
        return torch.sum(torch.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2))

    def set_params(self, params):
        """Set shape parameters from a dict of
        'param_name':(param, has_grad) entries"""

        self.params = {}
        for key, p in params.items():

            if key == "points":
                p = convert_param(p, scale=Scene.scale)
                # print('Setting points',  Scene.scale, p.max())
                self.params[key] = p
            else:
                p = convert_param(p)
                self.params[key] = p

    def param(self, name, numpy=False):
        """Get actual tensor from paramters"""
        if False:  # name=='points':
            # print('points', Scene.scale)
            # print(self.params[name].max())
            return self.params[name] * Scene.scale
        if numpy:
            return self.params[name].detach().cpu().numpy()
        return self.params[name]

    def has_grad(self, name):
        """Returns true if a parameter exists and it has gradients"""
        if not name in self.params:
            return False
        if self.params[name] is None:
            return False
        return self.params[name].requires_grad

    def setup(self):
        pass

    def update(self):
        pass

    def postprocess(self, requires_grad=False):
        pass

    def get_points(self):
        return self.param("points") * Scene.scale

    def get_stroke_width(self):
        return self.param("stroke_width")

    def shape_to_canvas(self):
        return None

    def to_dict(self):
        d = {"degree": self.degree}
        d.update(
            {
                key: val.detach().cpu().numpy()
                for key, val in self.params.items()
                if val is not None
            }
        )
        return d


def det22(mat):
    return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]


def args_to_params(**kwargs):
    params = {}
    for key, value in kwargs.items():
        if type(value) == tuple:
            params[key] = value
        else:
            params[key] = (value, False)
    return params


class Transform:
    def __init__(self, pos, rot, scale):
        self.pos = convert_param(pos)
        self.rot = convert_param(rot)
        self.scale = convert_param(scale)

    def shape_to_canvas(self):
        ct = torch.cos(self.rot)
        st = torch.sin(self.rot)

        rot = torch.vstack([torch.cat([ct, -st]), torch.cat([st, ct])])
        scale = torch.diag(
            self.scale * torch.ones(2, device=device, dtype=torch.float32)
        )

        self.transform = torch.cat(
            [
                torch.cat([torch.matmul(rot, scale), self.pos.reshape(-1, 1)], axis=1),
                torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=device),
            ],
            axis=0,
        )
        return self.transform


class OrientedBox(Shape):
    """Ellipse, does not handle stroke color"""

    def __init__(self, **kwargs):
        params = args_to_params(**kwargs)

        _params = {
            "center": ([0.0, 0.0], False),
            "radius": ([10.0, 5.0], False),
            "rot": ([0.0], False),
            "scale": ([1.0, 1.0], False),
            "pos": ([0.0, 0.0], False),
            "stroke_width": (1.0, False),
        }

        _params.update(params)

        # if _params['transform'][0].shape[0] > 2:
        #     _params['transform'] = (_params['transform'][0][:2,:], _params['transform'][1])

        self.set_params(_params)
        self.primitives = [self.build_box()]

    def build_box(self):
        p = self.param("center")
        r = self.param("radius")
        points = torch.vstack(
            [
                p + torch.tensor([-r[0], -r[1]]),
                p + torch.tensor([r[0], -r[1]]),
                p + torch.tensor([r[0], r[1]]),
                p + torch.tensor([-r[0], r[1]]),
            ]
        )
        return pydiffvg.Polygon(
            points=points, stroke_width=self.param("stroke_width"), is_closed=True
        )

    def shape_to_canvas(self):
        theta = self.param("rot")
        ct = torch.cos(theta)
        st = torch.sin(theta)

        rot = torch.vstack([torch.cat([ct, -st]), torch.cat([st, ct])])
        s = self.param("scale")
        scale = torch.diag(
            s
        )  # torch.cat([s[0:1], s[0:1]*s[1:2]])) # self.param('scale'))

        self.transform = torch.cat(
            [
                torch.cat(
                    [torch.matmul(rot, scale), self.param("pos").reshape(-1, 1)], axis=1
                ),
                torch.tensor([[0.0, 0.0, 1.0]]),
            ],
            axis=0,
        )
        return self.transform

    def update(self):
        self.primitives[0].stroke_width = self.param("stroke_width") / torch.sqrt(
            det22(self.transform)
        )


class Ellipse(Shape):
    """Ellipse, does not handle stroke color, Bezier approx"""

    def __init__(self, **kwargs):
        params = args_to_params(**kwargs)

        _params = {
            "center": ([0.0, 0.0], False),
            "radius": ([1.0, 1.0], False),
            "rot": ([0.0], False),
            "scale": ([1.0, 1.0], False),
            "pos": ([0.0, 0.0], False),
            "stroke_width": (1.0, False),
        }

        _params.update(params)

        self.set_params(_params)
        self.primitives = [self.build_ellipse()]
        self.transform = torch.eye(3).to(device)

    def build_ellipse(self):
        p = self.param("center")
        r = self.param("radius")
        points = torch.vstack(
            [
                p + torch.tensor([-r[0], -r[1]], device=device),
                p + torch.tensor([r[0], -r[1]], device=device),
                p + torch.tensor([r[0], r[1]], device=device),
                p + torch.tensor([-r[0], r[1]], device=device),
            ]
        )

        degree = 3
        num_segments = 4  # bezier.num_bezier(4, degree)
        num_control_points = (
            torch.zeros(num_segments, dtype=torch.int32, device=device) + degree - 1
        )
        points = cubic_bspline(points, periodic=True)
        return pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=self.param("stroke_width"),
            is_closed=True,
        )

    def shape_to_canvas(self):
        return self.transform

    def update(self):
        theta = self.param("rot")
        ct = torch.cos(theta)
        st = torch.sin(theta)

        rot = torch.vstack([torch.cat([ct, -st]), torch.cat([st, ct])])
        s = self.param("scale")

        scale = torch.diag(
            s
        )  # torch.cat([s[0:1], s[0:1]*s[1:2]])) # self.param('scale'))

        self.transform = torch.cat(
            [
                torch.cat(
                    [torch.matmul(rot, scale), self.param("pos").reshape(-1, 1)], axis=1
                ),
                torch.tensor([[0.0, 0.0, 1.0]], device=device),
            ],
            axis=0,
        )  # .to(device)

        self.primitives[0].stroke_width = self.param("stroke_width") / torch.sqrt(
            det22(self.transform)
        )
        self.primitives[0].shape_to_canvas = self.transform
        pass  #
        # if self.has_grad('center'):
        #     #print('updating points')
        #     self.primitives[0].center = self.param('center')
        # if self.has_grad('radius'):
        #     self.primitives[0].radius = self.param('radius')
        # # shape_to_canvas updated globally


class Polygon(Shape):
    def __init__(self, points=None, closed=False, postprocess=None, **kwargs):
        super().__init__()
        params = {}
        if points is not None:
            params["points"] = (points, True)
        params.update(args_to_params(**kwargs))
        _params = {"stroke_width": (1.0, False)}

        _params.update(params)
        self.set_params(_params)

        # Add postprocess method if specified
        if postprocess is not None:
            setattr(Polygon, "postprocess", postprocess)

        self.degree = 1
        self.closed = closed
        self.postprocess(False)
        self.setup()
        self.primitives = [
            pydiffvg.Polygon(
                points=self.get_points(),  # param('points'),
                stroke_width=self.get_stroke_width(),  # param('stroke_width'),
                is_closed=closed,
            )
        ]

    def update(self):
        self.setup()
        if self.has_grad("points"):
            # print('updating points')
            pts = self.get_points()
            if self.transform is not None:
                mat = self.transform.transform

                pts = (
                    (
                        mat
                        @ torch.hstack(
                            [
                                pts,
                                torch.ones(
                                    (len(pts), 1), device=device, dtype=torch.float32
                                ),
                            ]
                        ).T
                    )
                    .T[:, :2]
                    .contiguous()
                )
            self.primitives[0].points = pts  # self.get_points() #self.param('points')
        if self.has_grad("stroke_width"):
            self.primitives[
                0
            ].stroke_width = self.get_stroke_width()  # self.param('stroke_width')


class Path(Shape):
    """Generic path, can be used to construct a piecewise cubic path
    NB for closed curves last control point is first"""

    def __init__(
        self,
        points=None,
        degree=1,
        closed=False,
        use_distance_approx=False,
        postprocess=None,
        split_pieces=False,
        scale=None,
        **kwargs,
    ):
        super().__init__()

        if scale is None:
            scale = Scene.scale

        params = {}

        if points is not None:
            params["points"] = (points, True)

        params.update(args_to_params(**kwargs))

        _params = {"stroke_width": (1.0, False)}

        _params.update(params)
        self.set_params(_params)

        # if 'stroke_width' in params and not geom.is_number(params['stroke_width']):

        self.degree = degree
        self.closed = closed

        # Add postprocess method if specified
        if postprocess is not None:
            setattr(Path, "postprocess", postprocess)

        # Apply postprocess to points before get_points, which might construct bezier segments
        self.postprocess(False)

        self.setup()

        # NB this can be overridden by subclasses to create appropriate number of control points
        points = self.get_points()
        num_segments = self.num_segments(points)
        # It is annoying to count the number of control points for different derived classes
        # so if the user provides a list with a single element we assume it is the correct length
        sw = self.params["stroke_width"]
        if len(sw.shape) == 1 and len(sw) == 1:
            self.params["stroke_width"] = torch.tensor(
                torch.ones(points.shape[0]).to(device) * sw,
                requires_grad=sw.requires_grad,
            ).to(device)

        num_control_points = (
            torch.zeros(num_segments, dtype=torch.int32) + self.degree - 1
        )
        self.split_pieces = split_pieces
        if split_pieces:
            w = self.get_stroke_width()
            print("wshape", w.shape)
            nc = self.degree

            # if closed:
            #    points = torch.vstack([points, points[0]])
            #    w = torch.cat([w, w[0:1]])
            # print('wshape', w.shape)
            # print('pshape', points.shape)
            self.primitives = []
            for i in range(num_segments):
                ctrl = num_control_points[i : i + 1]
                pts = points[i * nc : i * nc + nc + 1]
                ww = w[i * nc : i * nc + nc + 1]

                # print(ctrl.shape, pts.shape, ww.shape)
                prim = pydiffvg.Path(
                    num_control_points=ctrl,
                    points=pts,
                    stroke_width=ww,
                    is_closed=False,
                    use_distance_approx=use_distance_approx,
                )
                self.primitives.append(prim)
        else:
            self.primitives = [
                pydiffvg.Path(
                    num_control_points=num_control_points,
                    points=points,
                    stroke_width=self.get_stroke_width(),  # Get potentially processed stroke width
                    is_closed=closed,
                    use_distance_approx=use_distance_approx,
                )
            ]

    def num_segments(self, points):
        n = len(points)

        if self.closed:
            return bezier.num_bezier(n + 1, self.degree)
        return bezier.num_bezier(n, self.degree)

    def num_points(self):
        return len(self.param("points"))

    def has_varying_width(self):
        w = self.param("stroke_width")
        return len(w.shape) > 0 and len(w) > 1 and self.width_func is None

    def get_beziers(self):
        points = self.primitives[0].points
        if self.closed:
            points = torch.vstack([points, points[0]])
        return bezier.chain_to_beziers(points, self.degree)

    def has_varying_width(self):
        w = self.param("stroke_width")
        return len(w.shape) > 0 and len(w) > 1

    def samples(self, num_or_u, der=0, thick=True):
        return self.bezier_samples(num_or_u, der, thick)

    def domain(self):
        return 0.0, 1.0

    def bezier_samples(self, num_or_u, der=0, thick=True):
        Cp = self.get_points()
        num = bezier.num_bezier(Cp.shape[0], self.degree)  # degree)
        if thick and self.has_varying_width():
            Cp = torch.hstack([Cp, self.get_stroke_width().reshape(-1, 1)])
        if self.closed:
            Cp = torch.vstack([Cp, Cp[0]])
        if type(num_or_u) == int:
            subd = num_or_u // num
            t = np.linspace(0, 1.0, subd)  # , device=device, dtype=Cp.dtype)
        else:
            t = num_or_u
            if t[-1] > 1:
                print("Warning, bezier samples computed per piece and parameter > 1")

        B = torch.tensor(
            bezier.bezier_mat(self.degree, t, deriv=der), device=device, dtype=Cp.dtype
        )
        # B = torch.vstack([bezier.bernstein(self.degree, i)(t) for i in range(self.degree+1)])

        X = []
        for i in range(num):
            P = Cp[i * self.degree : i * self.degree + self.degree + 1, :]
            Y = (P.T @ B).T
            X += [Y[:-1]]
        X.append(Cp[-1])
        return torch.vstack(X)

    def get_points(self):
        return self.param("points")

    def get_stroke_width(self):
        return self.param("stroke_width")

    def refresh(self):
        # Update in case we change number of points dynamically
        points = self.get_points().detach()
        num_segments = self.num_segments(points)
        num_control_points = (
            torch.zeros(num_segments, dtype=torch.int32, device=device)
            + self.degree
            - 1
        )

        if len(self.primitives) > 1:
            print("Control point update not supported with multi-primitives")
            # pass
        else:
            self.primitives[0].num_control_points = num_control_points

    def update(self):
        self.setup()
        self.refresh()

        pts = self.get_points()
        if self.transform is not None:
            mat = self.transform.transform
            pts = (
                (
                    mat
                    @ torch.hstack(
                        [
                            pts,
                            torch.ones(
                                (len(pts), 1), device=device, dtype=torch.float32
                            ),
                        ]
                    ).T
                )
                .T[:, :2]
                .contiguous()
            )

        num_segs = self.num_segments(pts)
        if self.has_grad("points"):

            nc = self.degree

            if self.split_pieces:
                for i in range(num_segs):
                    self.primitives[i].points = pts[i * nc : i * nc + nc + 1]
            else:
                self.primitives[0].points = pts
            # if pts.shape[1] == 2:
            #     self.primitives[0].points = pts
            # else:
            #     self.primitives[0].points = pts[:,:2]
            #     self.primitives[0].stroke_width = pts[:,2]
        if self.has_grad("stroke_width"):
            ww = self.get_stroke_width()
            if self.split_pieces:
                for i in range(num_segs):
                    self.primitives[i].stroke_width = ww[i * nc : i * nc + nc + 1]
            else:
                self.primitives[0].stroke_width = ww


class CubicBSpline(Path):
    """Constructs a piecewise cubic path using Bspline control points"""

    def __init__(
        self,
        points=None,
        tension=0.5,
        closed=False,
        use_distance_approx=False,
        postprocess=None,
        **kwargs,
    ):
        self.tension = tension
        super().__init__(
            points,
            degree=3,
            closed=closed,
            use_distance_approx=use_distance_approx,
            postprocess=postprocess,
            **kwargs,
        )

    def get_points(self):
        pts = cubic_bspline(self.param("points"), self.closed)
        return pts


def cardinal_spline(Q, c, closed=False):
    """Cardinal spline interpolation for a sequence of values"""
    isnp = isinstance(Q, np.ndarray)

    if closed:
        if isnp:
            Q = np.vstack([Q, Q[0:1]])
        else:
            Q = torch.concat([Q, Q[0:1]])
    n = len(Q)
    D = []
    for k in range(1, n - 1):
        # Assuming uniform parametrisation here
        d = (1 - c) * (Q[k + 1] - Q[k - 1])
        D.append(d)
    if closed:
        d1 = dn = (1 - c) * (Q[1] - Q[-2])
    else:
        d1 = (1 - c) * (Q[1] - Q[0])
        dn = (1 - c) * (Q[-1] - Q[-2])
    D = [d1] + D + [dn]
    P = [Q[0]]
    for k in range(1, n):
        p1 = Q[k - 1] + D[k - 1] / 3
        p2 = Q[k] - D[k] / 3
        p3 = Q[k]
        P += [p1, p2, p3]

    if closed:
        P = P[:-1]
    if isnp:
        return np.vstack(P)
    return torch.vstack(P)


class CardinalSpline(Path):
    def __init__(self, points=None, closed=False, tension=0.5, **kwargs):
        self.tension = tension
        clean_args = {
            key: item
            for key, item in kwargs.items()
            if key != "degree" and key != "closed"
        }
        super().__init__(
            points, degree=3, closed=closed, split_pieces=False, **clean_args
        )

    def get_degree(self):
        return self.degree

    def has_varying_width(self):
        w = self.param("stroke_width")
        return len(w.shape) > 0 and len(w) > 1

    def setup(self):
        P = self.param("points")
        w = self.param("stroke_width")
        has_width = self.has_varying_width()  # and self.width_func is None

        if has_width:
            Pw = torch.hstack([P, w.unsqueeze(1)])
        else:
            Pw = P

        Cp3 = cardinal_spline(Pw, self.tension, self.closed)

        self.points = Cp3[:, :2].contiguous()
        if has_width:
            self.widths = Cp3[:, 2].contiguous()
        else:
            self.widths = w  # torch.ones(len(Cp3), device=P.device)*w

    def get_points(self):
        return self.points * Scene.scale

    def get_stroke_width(self):
        return self.widths

# Compute these only once for each degree
gramian_cache = {}

class SmoothingBSpline(Path):
    """Constructs a Bspline approximation of a given degree
    Internally this will be sampled and rendered as a sequence of linear segments"""

    def __init__(
        self,
        points=None,
        closed=False,
        subd=10,
        degree=5,
        multiplicity=1,
        pspline=False,
        pspline_weight=True,  # Hack to get similar weights for derivative and penalty smoothing
        clamping=True,
        width_func=None,
        split_pieces=False,
        clamped=True,
        deriv_order=3,
        init_smooth_params={},
        **kwargs,
    ):

        self.subd = subd
        self.spline_degree = degree
        self.deriv_order = deriv_order
        self.clamping = clamping
        self.multiplicity = multiplicity
        self.point_offsets = None
        self.width_func = width_func

        self.pspline_weight = pspline_weight
        self.pspline = pspline
        self.init_smooth_params = init_smooth_params
        self.clamped = clamped
        self.width_postprocess = lambda x: x
        super().__init__(
            points, degree=3, closed=closed, split_pieces=split_pieces, **kwargs
        )

    def get_degree(self):
        return self.spline_degree

    def has_varying_width(self):
        w = self.param("stroke_width")
        return len(w.shape) > 0 and len(w) > 1 and self.width_func is None

    def setup(self):
        P = self.param("points")
        mult = self.multiplicity
        # Compute knots and add control point multiplicity
        p = self.spline_degree
        w = self.width_postprocess(self.param("stroke_width"))
        has_width = self.has_varying_width()  # and self.width_func is None
        if has_width:
            Pw = torch.hstack([self.param("points"), w.unsqueeze(1)])
        else:
            Pw = P
        # Multiplicity (assume this is set externally,
        # since we want to optimize the locations of the control points
        # Pw = torch.kron(Pw, torch.ones((mult, 1), device=Pw.device))
        self.spline_points = Pw

        # Clamping or periodicity is done internally
        if self.closed:
            # half_offset = p // 2
            # offset_remainder = p - half_offset
            # Q = torch.vstack([Pw[-offset_remainder:], Pw, Pw[:half_offset]])
            Q = torch.vstack([P[-1:], P, P[:p-1]])
        elif self.clamped:
            Q = torch.vstack([Pw[0]] * (p) + [Pw[mult:-mult]] + [Pw[-1]] * (p))
        else:
            Q = Pw
        self.Q = Q

        k = p + 1
        t, _, kt = bspline.tcu(
            P.detach().cpu().numpy(), k, mult, closed=self.closed, clamped=self.clamped
        )
        t = torch.tensor(t, device=device, dtype=P.dtype)
        self.kt = kt
        self.knots = t
        bezier_mat = torch.tensor(
            bspline.bspline_to_bezier_chain_mat(p, len(Q)),
            device=Q.device,
            dtype=Q.dtype,
        )
        Cp = bezier_mat @ Q
        if self.width_func is not None:
            Cp = torch.hstack([Cp, self.width_func(Cp).reshape(-1, 1)])
            has_width = True
        self.Cp = Cp

        if p > 3:
            b3mat = torch.tensor(
                bspline.bezier_chain_reduction_mat(p, 3, len(Cp)),
                device=Q.device,
                dtype=Q.dtype,
            )
            Cp3 = b3mat @ Cp
        else:
            Cp3 = Cp
        self.Cp3 = Cp3

        self.points = Cp3[:, :2].contiguous()
        if has_width:
            self.widths = Cp3[:, 2].contiguous()
        else:
            self.widths = w  # torch.ones(len(Cp3), device=P.device)*w

    def keypoint_times(self):
        return self.kt

    def domain(self):
        p = self.spline_degree
        k = p + 1
        return float(self.knots[p]), float(self.knots[-k])

    def spline_param_01(self, u):
        a, b = self.domain()
        return a + (b - a) * u

    def samples_with_bounds(self, max_speed, accel_time, dt=0.01, velocity=False):
        Q = self.Q.detach().cpu().numpy()
        deriv = 0
        X = bspline.resample_bspline(
            Q, dt, self.spline_degree, max_speed, accel_time, deriv=deriv
        )
        if velocity:
            # TODO use analytic
            dX = np.diff(X, axis=0) / dt
            return X, dX
        return X

    def samples(self, num_or_u, der=0, no_width=False, get_dt=False, numpy=False):
        dim = 3 if self.has_varying_width() else 2
        if no_width:
            dim = 2
        p = self.spline_degree
        k = p + 1
        n = len(self.Q)
        Q = self.Q[:, :dim]
        basis_knots = np.linspace(0, k, k + 1)
        bsp = bspline.BSpline.basis_element(basis_knots).derivative(der)
        Bk = lambda u: bsp(np.clip(u, 0, k))
        t = self.knots.detach().cpu().numpy()

        if type(num_or_u) == int:
            u = np.linspace(*self.domain(), num_or_u)
        elif geom.is_number(num_or_u):
            u = np.ones(1) * num_or_u
        else:
            u = num_or_u
        Bu = np.zeros((n, len(u)))
        for i in range(n):
            Bu[i, :] = Bk(u - t[i])
        Bu = torch.tensor(
            np.kron(Bu.T, np.eye(dim)), device=self.Q.device, dtype=self.Q.dtype
        )
        Qhat = Q.reshape(-1, 1)
        res = (Bu @ Qhat).reshape(len(u), dim)
        if numpy:
            res = res.detach().cpu().numpy()
        if get_dt:
            return res, u[1] - u[0]
        return res

    def inner(self, der, normalize=False, normalize_size=None):
        dim = 3 if self.has_varying_width() else 2
        n = len(self.Q)
        k = self.spline_degree + 1
        if (n, k, der) in gramian_cache:
            G = gramian_cache[(n, k, der)]
        else:
            # P-spline discrete approach
            if self.pspline:
                D = np.diff(np.eye(n), der)
                Gd = D @ D.T
                if self.pspline_weight:
                    G = bspline.uniform_gramian(n, k=k, der=der)
                    w = np.max(np.diag(G)) / np.max(np.diag(Gd))
                    Gd = Gd * w
                G = Gd
            else:
                # Analytic integral
                G = bspline.uniform_gramian(n, k=k, der=der)
            G = torch.tensor(
                np.kron(G, np.eye(dim)), device=self.Q.device, dtype=self.Q.dtype
            )
            gramian_cache[(n, k, der)] = G
        if normalize or normalize_size is not None:
            if normalize_size is None:
                P = self.param("points")
                D = torch.diff(P, axis=0)
                l = torch.sum(torch.sqrt(torch.sum(D**2, axis=1)))
            else:
                l = normalize_size
        else:
            l = 1.0
        t = self.knots
        Qhat = self.Q.reshape(-1, 1) / l  # (l**2) #(((t[-k]-t[k-1])**5)/l**2)

        res = Qhat.T @ G @ Qhat
        return res

    def get_points(self):
        return self.points * Scene.scale

    def get_stroke_width(self):
        return self.widths

    def to_dict(self):
        d = super().to_dict()
        d.update(
            {
                "degree": self.spline_degree,
                "closed": self.closed,
                "multiplicity": self.multiplicity,
                "pspline": self.pspline,
                "Cp3": self.Cp3.detach().cpu().numpy(),
                "Cp": self.Cp.detach().cpu().numpy(),
            }
        )
        return d


def to_tensor(v, dtype=torch.float32):
    return torch.tensor(v, dtype=dtype).contiguous().to(device)


def convert_param(p, dtype=torch.float32, scale=None):
    """Convert a (list, has_grad) param to a tensor with requires_grad set,
    allows for numpy/list inputs"""
    try:
        if p[0] is None:
            return None
        if False:  # scale is not None:
            var = to_tensor(p[0] / scale, dtype)
        else:
            var = to_tensor(p[0], dtype)
        if p[1]:
            var.requires_grad = True
    except TypeError:
        var = to_tensor(p, dtype)

    return var


def get_color_param(color, opacity=1.0):
    """Get RGBA color depending on cardinality of input, one component will use alpha (assuming black)"""
    if color is None:
        return None
    if len(color) == 1:
        if cfg.one_channel_is_alpha:
            return torch.concat([torch.zeros(3).to(device), color * opacity])
        else:

            return torch.concat(
                [torch.ones(3).to(device) * color, torch.ones(1).to(device) * opacity]
            )
    elif len(color) == 2:
        return torch.concat([torch.ones(3) * color[0], color[1:] * opacity])
    elif len(color) == 3:
        return torch.concat([color, torch.tensor([1.0 * opacity]).to(device)])
    return color


def postprocess_smooth(alpha):
    def smooth(self, requires_grad):
        path = self.params["points"]
        with torch.no_grad():
            path[:, 0] = (1 - alpha) * path[:, 0] + alpha * (
                (torch.roll(path[:, 0], 1) + torch.roll(path[:, 0], -1)) / 2.0
            )
            path[:, 1] = (1 - alpha) * path[:, 1] + alpha * (
                (torch.roll(path[:, 1], 1) + torch.roll(path[:, 1], -1)) / 2.0
            )

    return smooth


def postprocess_schematize_points(C, start_ang):
    """schematization postprocess on points (Polygon or Path)"""

    def schematize(self):
        self.params["points"].data = quantize_iterative_round(
            self.params["points"],
            C,
            start_ang,
            requires_grad=self.params["points"].requires_grad,
            closed=self.closed,
        )

    return schematize


def cubic_bspline(P, periodic=False):
    """Naive implementation of Bohm's algorithm for knot insertion"""
    def lerp(a, b, t):
        return a + t * (b - a)

    m = len(P)
    if periodic:
        P = torch.vstack([P[-1], P, P[0], P[1]])
    else:
        P = torch.vstack([P[0], P, P[-1]])  # P #
        # P = torch.vstack([P[0], P[0], P, P[-1], P[-1]])

    n = P.shape[0]
    Cp = []
    for i in range(n - 3):
        p = P[i : i + 4]
        b1 = lerp(p[1], p[2], 1.0 / 3)
        b2 = lerp(p[2], p[1], 1.0 / 3)
        l = lerp(p[1], p[0], 1.0 / 3)
        r = lerp(p[2], p[3], 1.0 / 3)

        if not Cp:
            b0 = lerp(l, b1, 0.5)
            b3 = lerp(b2, r, 0.5)
            Cp += [b0, b1, b2, b3]
        else:
            b3 = lerp(b2, r, 0.5)
            Cp += [b1, b2, b3]
    if periodic:  # Diffvg assumes last control point is fitst when curve is closed
        Cp = Cp[:-1]
    return torch.vstack(Cp)
