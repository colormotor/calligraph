#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
from . import diffvg_utils, geom, plut, config, fs
import torch.nn.functional as F
from numpy.linalg import norm
import pdb

device = config.device
dtype = torch.float32

log_eps = 1e-5
delta_sensitivity = 1e-3


def npy(t):
    return t.detach().cpu().numpy()


def tens(v):
    return torch.tensor(v, device=device, dtype=dtype)


def bezier_chain(X, dX):
    dim = X.shape[1]
    D = (dX) / 3
    P1 = X[:-1] + D[:-1]
    P2 = X[1:] - D[1:]
    Cp = torch.vstack((X[0:1], torch.stack((P1, P2, X[1:]), dim=1).reshape(-1, dim)))
    return Cp


class SLMTrajectory(diffvg_utils.Path):
    """Sigma-lognormal trajectory"""

    def __init__(
        self,
        points,
        Delta_t,
        # Ac=0.001,
        # T=0.8,
        subd=5,
        **kwargs
    ):

        m = len(Delta_t)
        params = {
            "Delta_t": (Delta_t, True),
            "delta": (np.zeros(m), False),
            "Ac": (np.ones(m) * 0.001, False),
            "T": (np.ones(m) * 0.8, False),
        }
        params.update(diffvg_utils.args_to_params(**kwargs))
        self.width_func = None
        self.subd = subd
        super().__init__(
            points,
            degree=3,
            closed=False,
            split_pieces=False,
            **diffvg_utils.skip_kwargs(["degree", "closed"], params)
        )  # kwargs)

    def has_varying_width(self):
        w = self.param("stroke_width")

        return len(w.shape) > 0 and len(w) > 1 and self.width_func is None

    def setup(self):
        P = self.param("points")
        m = len(P) - 1
        Delta_t = self.param("Delta_t")

        Ac = torch.ones(m, device=device, dtype=dtype) * self.param("Ac")
        T = torch.ones(m, device=device, dtype=dtype) * self.param("T")
        delta = torch.ones(m, device=device, dtype=dtype) * self.param("delta")

        has_width = self.has_varying_width()
        if has_width:
            w = self.param("stroke_width")
            Pw = torch.hstack([self.param("points"), w.unsqueeze(1)])
        else:
            Pw = P
        X, dX, ddX, endt, strokes = wslm_kinematics(
            Pw, Delta_t, Ac, T, delta, m * self.subd, get_strokes=True
        )
        self.strokes = strokes
        Cp = bezier_chain(X, dX)  # .contiguous()
        self.points = Cp[:, :2].contiguous()  # bezier_chain(X, dX).contiguous()
        if has_width:
            self.widths = torch.relu(Cp[:, 2].contiguous())
        else:
            self.widths = self.param("stroke_width")
        self.endt = endt
        self.endt_norm = endt / m

    def samples(
        self,
        subd,
        get_strokes=False,
        velocity=False,
        acceleration=False,
        delta_t_mul=1.0,
    ):
        P = self.param("points")
        m = len(P) - 1
        Delta_t = self.param("Delta_t") * delta_t_mul

        Ac = torch.ones(m, device=device, dtype=dtype) * self.param("Ac")
        T = torch.ones(m, device=device, dtype=dtype) * self.param("T")
        delta = torch.ones(m, device=device, dtype=dtype) * self.param("delta")

        if self.has_varying_width():
            P = torch.vstack([P.T, self.param("stroke_width")]).T
        X, dX, ddX, endt, strokes = wslm_kinematics(
            P, Delta_t, Ac, T, delta, m * subd, get_strokes=True
        )
        if get_strokes:
            if velocity:
                return X, dX, strokes
            if acceleration:
                return X, dX, ddX, strokes
            return X, strokes
        if velocity:
            return X, dX
        return X

    def samples_with_bounds(
        self, max_speed, accel_time, dt=0.01, velocity=False
    ):  # initial_subd):
        P = self.param("points")
        m = len(P) - 1
        Delta_t = self.param("Delta_t")

        Ac = torch.ones(m, device=device, dtype=dtype) * self.param("Ac")
        T = torch.ones(m, device=device, dtype=dtype) * self.param("T")
        delta = torch.ones(m, device=device, dtype=dtype) * self.param("delta")

        if self.has_varying_width():
            P = torch.vstack([P.T, self.param("stroke_width")]).T
        # Hacky dt calc
        num_steps = m * 10
        X, dX, ddX, endt, strokes = wslm_kinematics(
            P, Delta_t, Ac, T, delta, num_steps, get_strokes=True
        )
        num_steps = int((endt / dt).detach().cpu().numpy()) + 1
        X, dX, ddX, endt, strokes = wslm_kinematics(
            P, Delta_t, Ac, T, delta, num_steps, get_strokes=True
        )
        smax = torch.norm(dX / dt, dim=1).max()
        amax = torch.norm(ddX, dim=1).max()
        max_accel = max_speed / accel_time
        div_s = max_speed / smax
        div_a = torch.sqrt(max_accel / amax)
        div = min(div_s, div_a)
        dt = dt * div
        num_steps = int((endt / dt).detach().cpu().numpy()) + 1
        X, dX, ddX, endt, strokes = wslm_kinematics(
            P, Delta_t, Ac, T, delta, num_steps, get_strokes=True
        )
        if velocity:
            return X, dX
        return X

    def get_points(self):
        return self.points

    def get_stroke_width(self):
        return self.widths

    def time_cost(self, reg=0):
        loss = 0.0
        if reg > 0:
            # Isochrony
            Delta_t = self.param("Delta_t")
            if len(Delta_t) > 2:
                loss = torch.var(Delta_t) * reg
            # print('delta t var: ', loss)
            # print('endt_norm:', self.endt_norm)
        return loss + self.endt_norm

    def jerk(self, subd, ref_size=1):
        P = self.param("points")
        m = len(P) - 1
        Delta_t = self.param("Delta_t")
        Ac = torch.ones(m, device=device, dtype=dtype) * self.param("Ac")
        T = torch.ones(m, device=device, dtype=dtype) * self.param("T")
        delta = torch.ones(m, device=device, dtype=dtype) * self.param("delta")
        _, _, ddX, endt = wslm_kinematics(P / ref_size, Delta_t, Ac, T, delta, m * subd)
        dt = endt / m * subd
        dddX = torch.diff(ddX, axis=0) / endt  # /dt
        n = len(ddX)
        return torch.sum(dddX[:, 0] ** 2 + dddX[:, 1] ** 2) / n


diffvg_utils.shape_classes["SLMTrajectory"] = SLMTrajectory


def trajectories_and_speeds_from_scene(path, delta_t_mul=1.0, subd=15, dt=0.1):
    strokes = fs.load_json(path)

    trajectories = []
    speeds = []

    for group in strokes["groups"]:
        for shape in group["shapes"]:
            m = len(shape["points"]) - 1
            P = np.array(shape["points"])
            # P = geom.tsm(mat, P)
            Delta_t = np.array(shape["Delta_t"])
            path = SLMTrajectory(
                points=P,
                Delta_t=Delta_t * delta_t_mul,  # shape['Delta_t'],
                Ac=(shape["Ac"], False),
                delta=(shape["delta"], False),
            )
            endt = path.endt.detach().cpu()
            n = int(endt / dt)
            samples, dX, ssss = path.samples(n, get_strokes=True, velocity=True)
            dX = dX.detach().cpu().numpy()
            samples = samples.detach().cpu().numpy()
            speeds.append(np.sqrt(dX[:, 0] ** 2 + dX[:, 1] ** 2))
            trajectories.append(samples)

    return trajectories, speeds


def lognormal(x, x0, mu, sigma, eps=1e-5):
    """3 parameter lognormal"""
    x = x - x0
    x = F.relu(x - eps) + eps
    y = (
        1
        / (((x) * np.sqrt(2 * np.pi) * sigma))
        * torch.exp(-((torch.log(x) - mu) ** 2) / (2 * sigma**2))
    )
    return y


def lognormal_prime(x, x0, mu, sigma):
    """Derivative of the lognormal function"""
    x = x - x0
    x = F.relu(x - log_eps) + log_eps
    l = (
        1
        / (((x) * np.sqrt(2 * np.pi) * sigma))
        * torch.exp(-((torch.log(x) - mu) ** 2) / (2 * sigma**2))
    )
    return l * (mu - sigma**2 - torch.log(x)) / (sigma**2 * x)


def lognormal_weight(t, t0, mu, sigma, eps=1e-5):
    """Lognormal interpolation between a and b"""
    t = t - t0
    t = F.relu(t - eps) + eps
    return 0.5 * (1 + torch.erf((torch.log(t) - mu) / (np.sqrt(2) * sigma)))


def mu_sigma(Ac, T):
    """Compute mu and sigma given profile asymmetry alpha and duration d"""
    if isinstance(Ac, torch.Tensor):
        tp = torch
    else:
        tp = np
    sigma = tp.sqrt(-tp.log(1.0 - Ac))
    mu = 3.0 * sigma - tp.log((-1.0 + tp.exp(6.0 * sigma)) / T)
    return mu, sigma


def lognormal_onset(mu, sigma):
    if isinstance(mu, torch.Tensor):
        tp = torch
    else:
        tp = np
    return tp.exp(mu - sigma * 3)


def compute_stroke_parameters(P, Delta_t, Ac, T, delta, t_offset=0.0):
    m = P.shape[0] - 1
    dim = P.shape[1]

    # make sure we have values for each of m strokes
    Delta_t = torch.ones(m, device=device, dtype=dtype) * Delta_t
    # Delta_t = torch.relu(torch.ones(m, device=device, dtype=dtype)*Delta_t)
    Ac = torch.ones(m, device=device, dtype=dtype) * Ac
    T = torch.ones(m, device=device, dtype=dtype) * T
    delta = torch.ones(m, device=device, dtype=dtype) * delta

    D = torch.diff(P, axis=0)
    mu, sigma = mu_sigma(Ac, T)

    # compute t0 values from delta
    # add small time offset ad the start to guarantee
    # that first stroke starts with zero velocity
    t0 = torch.zeros(m, device=device, dtype=dtype) + t_offset  # +0.05
    t0 = Delta_t[1:] * T[1:]
    t0 = torch.cumsum(
        torch.cat([torch.ones(1, device=device, dtype=dtype) * t_offset, t0]), dim=0
    )

    # Add onsets in order to shift lognormal to start
    t0 = t0 - torch.exp(mu[0] - sigma[0] * 3)

    return (t0, D, delta, mu, sigma)


def wslm_stroke_kinematics(t, t_0, mu, sigma, d, delta):
    w_t = lognormal_weight(t, t_0, mu, sigma)
    l = lognormal(t, t_0, mu, sigma)
    dl = lognormal_prime(t, t_0, mu, sigma)  # *D

    h = 1 / torch.sinc(delta / (2 * np.pi))

    dim = len(d)

    def R(theta):
        # Create a batch of N rot matrices
        ct = torch.cos(theta)
        st = torch.sin(theta)

        # shape: (N, 3, 3)
        return torch.stack(
            [
                torch.stack([ct, -st, torch.zeros_like(ct)], dim=-1),
                torch.stack([st, ct, torch.zeros_like(ct)], dim=-1),
                torch.stack(
                    [torch.zeros_like(ct), torch.zeros_like(ct), torch.ones_like(ct)],
                    dim=-1,
                ),
            ],
            dim=1,
        )[:, :dim, :dim]

    def dR(theta):
        # Create a batch of N rot matrices
        ct = delta * torch.cos(theta) * l
        st = delta * torch.sin(theta) * l

        # shape: (N, 3, 3)
        return torch.stack(
            [
                torch.stack([-st, -ct, torch.zeros_like(ct)], dim=-1),
                torch.stack([ct, -st, torch.zeros_like(ct)], dim=-1),
                torch.stack(
                    [torch.zeros_like(ct), torch.zeros_like(ct), torch.zeros_like(ct)],
                    dim=-1,
                ),
            ],
            dim=1,
        )[:, :dim, :dim]

    H = torch.eye(dim, device=device, dtype=torch.float32)
    H[:2, :2] *= h
    M = (1 / (2 * torch.tan(delta / 2))) * torch.tensor(
        [[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32, device=device
    )[:dim, :dim]

    # Prepare for batch
    d = d.reshape(-1, 1)  # unsqueeze(-1) #0).unsqueeze(-1)
    # # Displacement
    if abs(delta) < delta_sensitivity:
        x = (w_t * d).T  # .squeeze(0).T
    else:
        theta = -delta + delta * w_t
        Md = M @ d
        Zd = (
            torch.vstack(
                [0.5 * torch.ones_like(w_t), 0.5 * torch.ones_like(w_t), w_t - 0.5]
            )[:dim, :]
            * d
        )
        x = (R(theta) @ (d / 2 - Md)).squeeze(-1) + Md.T + Zd.T

    # Velocity
    theta = -delta / 2 + delta * w_t
    HRD = (H @ R(theta) @ d).squeeze(-1)
    vel = (HRD.T * l).T

    # Acceleration
    acc = (((H @ dR(theta) @ d).squeeze(-1).T * l) + (HRD.T * dl)).T

    return x, vel, acc


def wslm_kinematics_raw(
    num_steps,
    x0,
    t0,
    D,
    delta,
    mu,
    sigma,
    full_output=False,
    get_weights=False,
    t_offset=0.0,
    get_strokes=False,
    pos_only=False,
    start_t=0.0,
    dt=0.0,
):
    # endtime
    endt = t0[-1] + torch.exp(mu[-1] + sigma[-1] * 3)
    # endt = T*(len(P)-1)
    # time steps
    if dt > 0:
        t = torch.arange(start_t, endt, dt).to(device)
    else:
        t = torch.linspace(start_t, endt, num_steps).to(
            device
        )  # *endt #, requires_grad=True)
        dt = endt / (num_steps - 1)
    # t = torch.arange(0.0, endt, dt)
    n = len(t)
    m = len(delta)
    dim = D.shape[1]
    X = torch.ones((n, dim), device=device, dtype=dtype) * x0
    dX = torch.zeros((n, dim), device=device, dtype=dtype)
    ddX = torch.zeros((n, dim), device=device, dtype=dtype)  # Invalid for 3d

    strokes = []
    for i in range(m):
        d, vel, acc = wslm_stroke_kinematics(
            t, t0[i], mu[i], sigma[i], D[i, :], delta[i]
        )
        X += d
        dX += vel  # *dt
        ddX += acc
        strokes.append(vel.detach().cpu().numpy())
    if pos_only:
        return X
    if get_strokes:
        return X, dX * dt, ddX, endt, strokes
    return X, dX * dt, ddX, endt


def wslm_kinematics(
    P,
    Delta_t,
    Ac,
    T,
    delta,
    num_steps,
    full_output=True,
    get_weights=False,
    t_offset=0.0,
    get_strokes=False,
):
    """Weighted parameterisation of a sigma-lognormal trajectory
    using alpha ]0,1] as an assymetry parameter,
    and d for the duration of the lognormal"""
    t0, D, delta, mu, sigma = compute_stroke_parameters(
        P, Delta_t, Ac, T, delta, t_offset
    )
    return wslm_kinematics_raw(
        num_steps,
        P[0],
        t0,
        D,
        delta,
        mu,
        sigma,
        full_output,
        get_weights,
        t_offset,
        get_strokes,
    )


def action_plan_arcs(Vp, Theta, subd=100, concat=False):
    """Return points of sigma-lognormal action plan arcs"""
    m = Vp.shape[1] - 1

    if geom.is_number(Theta):
        Theta = np.ones(m) * Theta

    arcs = []

    for a, b, theta in zip(Vp.T, Vp[:, 1:].T, Theta):
        mp = a + (b - a) * 0.5

        if abs(theta) < 1e-9:
            theta = 1e-9

        d = b - a
        l = norm(d)
        r = l / (-np.sin(theta) * 2)

        h = (1 - np.cos(theta)) * r
        h2 = r - h
        p = np.dot([[0, -1], [1, 0]], d)
        p = p / norm(p)

        cenp = mp - p * h2
        theta_start = np.arctan2(p[1], p[0])
        A = np.linspace(theta_start - theta, theta_start + theta, subd)
        arc = (
            np.tile(cenp.reshape(-1, 1), (1, subd))
            + np.vstack([np.cos(A), np.sin(A)]) * r
        )
        arcs.append(arc)

    if concat:
        return np.hstack(arcs)
    else:
        return arcs


def plot_action_plan(
    Vp,
    Theta,
    label="Virtual targets",
    arc_label="Arcs",
    markersize=1,
    clr="r",
    linewidth=1,
):
    import matplotlib.pyplot as plt

    subd = 100
    Vp = Vp.T
    m = Vp.shape[1] - 1

    if geom.is_number(Theta):
        Theta = np.ones(m) * Theta
    else:
        Theta = np.array(Theta)  # assure it is an array
    Theta = Theta * 0.5

    if label:
        lbl = arc_label
    else:
        lbl = ""
    for a, b, theta in zip(Vp.T, Vp[:, 1:].T, Theta):
        mp = a + (b - a) * 0.5

        if abs(theta) < 1e-9:
            theta = 1e-9

        d = b - a
        l = norm(d)
        r = l / (-np.sin(theta) * 2)

        h = (1 - np.cos(theta)) * r
        h2 = r - h
        p = np.dot([[0, -1], [1, 0]], d)
        p = p / norm(p)

        cenp = mp - p * h2
        theta_start = np.arctan2(p[1], p[0])
        A = np.linspace(theta_start - theta, theta_start + theta, subd)
        arc = (
            np.tile(cenp.reshape(-1, 1), (1, subd))
            + np.vstack([np.cos(A), np.sin(A)]) * r
        )
        plt.plot(
            arc[0, :], arc[1, :], ":" + clr, alpha=0.6, label=lbl, linewidth=linewidth
        )
        lbl = ""

    plt.plot(Vp[0, :], Vp[1, :], clr + "o", label=label, markersize=markersize)


###########################
###########################


def compute_stroke_parameters_angle(P, Delta_t, Ac, T, delta, t_offset=0.0):
    m = P.shape[0] - 1
    dim = P.shape[1]

    # make sure we have values for each of m strokes
    Delta_t = torch.ones(m, device=device, dtype=dtype) * Delta_t
    Ac = torch.ones(m, device=device, dtype=dtype) * Ac
    T = torch.ones(m, device=device, dtype=dtype) * T
    delta = torch.ones(m, device=device, dtype=dtype) * delta

    V = torch.diff(P[:, :2], axis=0)
    # D = V.norm(dim=1)
    D = torch.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    theta = torch.arctan2(V[:, 1], V[:, 0])
    if dim > 2:
        gamma = torch.diff(P[:, 2])
        # gamma = V[:,2] #torch.arcsin(V[:,-1]/D) # Only valid if 3d
    else:
        gamma = None
    mu, sigma = mu_sigma(Ac, T)

    # compute t0 values from delta
    # add small time offset ad the start to guarantee
    # that first stroke starts with zero velocity
    t0 = torch.zeros(m, device=device, dtype=dtype) + t_offset  # +0.05
    t0 = Delta_t[1:] * T[1:]
    t0 = torch.cumsum(
        torch.cat([torch.ones(1, device=device, dtype=dtype) * t_offset, t0]), dim=0
    )

    # Add onsets in order to shift lognormal to start
    t0 = t0 - torch.exp(mu[0] - sigma[0] * 3)

    return (t0, D, theta, delta, mu, sigma, gamma)


def wslm_stroke_kinematics_angle(t, t_0, mu, sigma, D, theta, delta, gamma=None):
    w_t = lognormal_weight(t, t_0, mu, sigma)
    l = lognormal(t, t_0, mu, sigma)
    dl = lognormal_prime(t, t_0, mu, sigma)  # *D
    ts = theta - delta / 2
    tf = theta + delta / 2
    phi = ts + (tf - ts) * w_t
    h = 1 / torch.sinc(delta / (2 * np.pi))
    theta_0 = theta - np.pi / 2 - delta / 2

    theta_0 = theta - np.pi / 2 - delta / 2
    threeD = False
    if gamma is not None:
        threeD = True
    # # Displacement
    if abs(delta) < delta_sensitivity:
        if threeD:
            d = torch.vstack(
                [
                    D * h * torch.cos(theta) * w_t,
                    D * h * torch.sin(theta) * w_t,
                    gamma * w_t,
                ]
            ).T
        else:
            d = (
                D
                * torch.vstack(
                    [h * torch.cos(theta) * w_t, h * torch.sin(theta) * w_t]
                ).T
            )
    else:
        if threeD:
            d = torch.vstack(
                [
                    D
                    * h
                    * (torch.cos(theta_0 + w_t * delta) - torch.cos(theta_0))
                    / delta,
                    D
                    * h
                    * (torch.sin(theta_0 + w_t * delta) - torch.sin(theta_0))
                    / delta,
                    gamma * w_t,
                ]
            ).T  # <- hack
        else:
            d = (
                D
                * torch.vstack(
                    [
                        h
                        * (torch.cos(theta_0 + w_t * delta) - torch.cos(theta_0))
                        / delta,
                        h
                        * (torch.sin(theta_0 + w_t * delta) - torch.sin(theta_0))
                        / delta,
                    ]
                ).T
            )  # <- hack
    # Velocity
    # Assume 2.5 d
    if threeD:
        vel = torch.vstack(
            [D * l * torch.cos(phi), D * l * torch.sin(phi), l * gamma]
        ).T
    else:
        vel = torch.vstack([D * l * torch.cos(phi), D * l * torch.sin(phi)]).T
    # Acc (invalid if 3d)
    acc = torch.vstack(
        [
            D * dl * torch.cos(phi) - D * delta * (l**2) * torch.sin(phi),
            D * dl * torch.sin(phi) + D * delta * (l**2) * torch.cos(phi),
        ]
    ).T

    return d, vel, acc


def wslm_kinematics_angle(
    P,
    Delta_t,
    Ac,
    T,
    delta,
    num_steps,
    full_output=True,
    get_weights=False,
    t_offset=0.0,
    get_strokes=False,
):
    """Weighted parameterisation of a sigma-lognormal trajectory
    using alpha ]0,1] as an assymetry parameter,
    and d for the duration of the lognormal"""
    t0, D, theta, delta, mu, sigma, gamma = compute_stroke_parameters(
        P, Delta_t, Ac, T, delta, t_offset
    )
    # endtime
    endt = t0[-1] + torch.exp(mu[-1] + sigma[-1] * 3)
    # time steps
    t = torch.linspace(0.0, 1, num_steps).to(device) * endt  # , requires_grad=True)
