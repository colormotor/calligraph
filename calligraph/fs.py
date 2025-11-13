#!/usr/bin/env python3
import json, os, sys
import numpy as np
import dataclasses


def load_json(path):
    import codecs

    try:
        with codecs.open(path, encoding="utf8") as fp:
            data = json.load(fp)
        return data
    except IOError as err:
        print(err)
        print("Unable to load json file:" + path)
        return {}


class CustomEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return json.JSONEncoder.default(self, obj)


def save_json(data, path, encoder=CustomEncoder):
    with open(path, "w") as fp:
        # default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        # The above breaks
        json.dump(data, fp, indent=4, sort_keys=True, cls=encoder)  # , default=default)


def create_dir(dir):
    """create a directory if it doesn't exist. silently fail if it already does exist"""
    try:
        os.makedirs(dir)
    except OSError:
        if not os.path.isdir(dir):
            raise OSError
    return dir


def files_in_dir(path, exts=[], nohidden=True):
    files = []
    path = os.path.expanduser(path)
    for file in os.listdir(path):
        if nohidden and os.path.basename(file).startswith("."):
            continue
        if exts:
            if type(exts) != list:
                exts = [exts]
            for ext in exts:
                if file.endswith(ext):
                    files.append(os.path.join(path, file))
                    break
        else:
            files.append(os.path.join(path, file))

    return files


def filename(path):
    name = os.path.basename(path)  # Gets the filename with extension
    name = os.path.splitext(name)[0]  # Removes the extension
    return name


def filename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def filename_ext(path):
    return os.path.splitext(os.path.basename(path))[1]


def download_file_once(url, local_path):
    import requests

    local_path = os.path.expanduser(local_path)

    # Check if the file already exists
    if os.path.exists(local_path):
        print(f"File already exists at {local_path}. Skipping download.")
    else:
        print(f"Downloading file from {url} to {local_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Save the file in chunks
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete.")

    return local_path


def load_gml(path, rotate=False, flip_vertical=False, get_timesteps=False):
    """Parse a GML (Graffiti markup language) file"""
    # f = open(path, 'r')
    data = load_json(path)  # json.loads(f.read())

    paths = data["gml"]["tag"]["drawing"]["stroke"]
    # tags = data['gml_keywords'] if 'gml_keywords' in data else 'none'
    # if 'txt' in data:
    #     txt = data['txt']
    #     print('Found text: ' + txt)
    # else:
    #     txt = 'unknown'
    txt = "unknown"
    tags = []
    # info = {'id':data['id'], 'tags':tags, 'txt':txt}

    S = []
    T = []
    if not paths:
        raise ValueError

    if type(paths) == dict:
        paths = [paths]

    for i, path in enumerate(paths):
        try:
            if rotate:
                P = [np.array([float(pt["y"]), -float(pt["x"])]) for pt in path["pt"]]
            else:
                P = [np.array([float(pt["x"]), float(pt["y"])]) for pt in path["pt"]]
            if get_timesteps:
                t = np.array([float(pt["t"]) for pt in path["pt"]])
                T.append(t)
            P = np.array(P).T
            if flip_vertical:
                P[1, :] = -P[1, :]
            S.append(P.T)
        except:
            print("corrupt path")

    if get_timesteps:
        return S, T
    return S
