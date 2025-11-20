# Python port of Discrete Curve Evolution
# Based on:
# L. J. Latecki and R. Lakaemper (1999): Convexity Rule for Shape Decomposition Based on Discrete Contour Evolution.
# X. Bai, L. J. Latecki, and W.-Y. Liu (2007): Skeleton Pruning by Contour Partitioning with Discrete Curve Evolution
# From https://sites.google.com/site/xiangbai/softwareforskeletonizationandskeletonpru

# RELEVANCE
# returns normalized relevance measure of required index
import numpy as np


def chord_lengths(P, closed=0):
    if closed:
        P = np.vstack([P, P[0]])
    D = np.diff(P, axis=0)
    L = np.sqrt(D[:, 0] ** 2 + D[:, 1] ** 2)
    return L


def acos(v):
    v = np.clip(v, -1, 1)
    ac = np.arccos(v)
    if np.isnan(ac):
        print("dce::acos, NaN in arccos call with " + str(v))
    return ac


def relevance(s, index, peri, keep):
    n = s.shape[0]

    if keep[index]:
        return np.inf

    if n < 3:
        print("dce::relevance, segment too short")

    # vertices
    i0 = index - 1
    i1 = index
    i2 = index + 1
    if i0 == -1:
        i0 = n - 1
    elif i2 >= n:
        i2 = 0

    # segments
    seg1x = s[i1, 0] - s[i0, 0]
    seg1y = s[i1, 1] - s[i0, 1]
    seg2x = s[i1, 0] - s[i2, 0]
    seg2y = s[i1, 1] - s[i2, 1]

    l1 = np.sqrt(seg1x * seg1x + seg1y * seg1y)
    l2 = np.sqrt(seg2x * seg2x + seg2y * seg2y)

    # turning angle (0-180)
    av = (np.dot(np.hstack([seg1x, seg1y]), np.vstack([seg2x, seg2y]))) / l1 / l2
    if np.isnan(av):
        return 0.0

    if np.isnan(av):
        print("dce::relevance, Nan in turning angle computation")
    a = 180 - acos(av) * 180 / np.pi

    # relevance measure
    v = a * l1 * l2 / (l1 + l2)
    v = v / peri  # normalize

    if np.isnan(v):
        print("Nan in dce::relevance")
    return v


def seglength(p1x, p1y, p2x, p2y):
    dx = p2x - p1x
    dy = p2y - p1y
    l = np.sqrt(dx * dx + dy * dy)
    return l


def blocked(s, i):
    # find neighbouring vertices
    n = s.shape[0]

    i0 = i - 1
    i1 = i + 1
    if i0 == -1:
        i0 = n - 1
    elif i1 >= n:
        i1 = 0

    # bounding box
    minx = np.min(np.hstack([s[i0, 0], s[i, 0], s[i1, 0]]))
    miny = np.min(np.hstack([s[i0, 1], s[i, 1], s[i1, 1]]))
    maxx = np.max(np.hstack([s[i0, 0], s[i, 0], s[i1, 0]]))
    maxy = np.max(np.hstack([s[i0, 1], s[i, 1], s[i1, 1]]))

    # check if any boundary-vertex is inside bounding box
    # first create index-set v=s\(i0,i,i1)
    if i0 < i1:
        k = [i1, i, i0]
    elif i0 > i:
        k = [i0, i1, i]
    elif i1 < i:
        k = [i, i0, i1]

    v = list(range(n))

    try:
        v.pop(k[0])
        v.pop(k[1])
        v.pop(k[2])
    except IndexError:
        print("ierr")

    b = 0

    for k in range(len(v)):
        px = s[v[k], 0]
        py = s[v[k], 1]

        # vertex px,py inside boundary-box ?
        if ((px < minx) or (py < miny) or (px > maxx) or (py > maxy)) == False:
            # inside, now test triangle
            a = s[i, :] - s[i0, :]  # a= i0 to i
            b = s[i1, :] - s[i, :]  # b= i to i1
            c = s[i0, :] - s[i1, :]  # c= i1 to i0

            e0 = s[i0, :] - np.hstack([px, py])
            e1 = s[i, :] - np.hstack([px, py])
            e2 = s[i1, :] - np.hstack([px, py])

            d0 = np.linalg.det(np.vstack([a, e0]))
            d1 = np.linalg.det(np.vstack([b, e1]))
            d2 = np.linalg.det(np.vstack([c, e2]))

            # INSIDE ?
            b = ((d0 > 0) and (d1 > 0) and (d2 > 0)) or (
                (d0 < 0) and (d1 < 0) and (d2 < 0)
            )
        if b:
            break

    return b


def dce(
    P,
    maxvalue,
    num=0,
    keep_end=1,
    closed=False,
    get_indices=False,
    keep=[],
    get_flags=False,
    perimeter=None,
):
    # EVOLUTION(slist, num,<maxvalue>,<keep_end>,<process_until_convex><show>)
    # discrete curve evolution of slist to n points
    # input: slist, num of vertices in resulting list
    #       optional: maxvalue: stop criterion,if >0.0 value overrides num
    #       optional: keep_end. if set to 1, endpoints are NOT deleted
    #       optional: process_until_convex. if set to 1, evolution continues until
    #                 shape is convex
    #       optional: show: displayflag
    # output: s=simplificated list
    #           value= vector containing values of remaining vertices in point-order
    #           delval= vector containing values of deleted vertices in order of deletion

    if num == 0:
        num = 2

    s = np.array(P)
    I = list(range(s.shape[0]))

    if maxvalue == 0.0:
        if get_flags:
            # Flags indicating wethere a vertex is removed
            return [1 for p in s]  # if i in I else 0 for i in range(P.shape[1])]
        if get_indices:
            return s, I
        return s

    delval = []
    if perimeter is None:

        peri = np.sum(chord_lengths(P, closed))
    else:

        peri = perimeter

    if closed:
        s = np.vstack([s, s[0]])
        I.append(0)

    n = s.shape[0]

    # Keep these vertices fixed
    if keep:
        keep = [k > 0 for k in keep]
    else:
        keep = [False for i in range(n)]

    if keep_end:
        keep[0] = keep[-1] = True

    # initialize value vector (containing the information value of each vertex)
    value = np.zeros(n)  # , 1);
    for i in range(n):
        value[i] = relevance(s, i, peri, keep)

    m = -1
    # mainloop
    while True:
        n = s.shape[0]
        if num >= n:
            break  # ready

        i = np.argmin(value)
        m = value[i]

        if m > maxvalue:  # ready
            break

        # test blocking
        if m > 0:
            bf = blocked(s, i)

            # if vertex is blocked, find first non-blocked vertex
            # this procedure is separated from the 'usual min-case'
            # for speed-reasons (sort is needed instead of min)
            if bf:
                ind = np.argsort(value)
                rel = value[ind]
                j = 1
                m = np.inf

                while j < n:
                    i = ind[j]
                    bf = blocked(s, i)
                    if bf == 0:
                        m = rel[j]
                        break

                    j = j + 1

                if m > maxvalue:
                    break  # ready.

        # delete vertex
        px = s[i, 0]
        py = s[i, 1]  # keep coordinates

        s = np.delete(s, i, 0)
        I = np.delete(I, i, 0)
        keep.pop(i)

        # s[i, :]=[]
        value = np.delete(value, i)
        # value[i]=[]
        delval.append(m)

        # neighbouring vertices
        i0 = i - 1
        i1 = i
        if i0 == -1:
            i0 = n - 1
        elif i1 >= n:
            i1 = 0

        # neighbouring vertices changed value
        value[i0] = relevance(s, i0, peri, keep)
        value[i1] = relevance(s, i1, peri, keep)

    if closed:
        P = P[:-1]
        s = s[:-1, :]
        I = I[:-1]

    if get_flags:
        # Flags indicating wethere a vertex is removed
        flags = [1 if i in I else 0 for i in range(P.shape[0])]
        return flags

    if get_indices:
        return s, I
    return s
