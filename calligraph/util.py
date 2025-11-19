#!/usr/bin/env python3
'''
 )   ___
(__/_____)     /) /) ,                 /)
  /       _   // //    _   __  _  __  (/
 /       (_(_(/_(/__(_(_/_/ (_(_(_/_)_/ )_
(______)             .-/       .-/
                    (_/       (_/


'''

import torch
import numpy as np
import os, random, pdb, time, sys, yaml, json, argparse, ast
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import config
import shutil


class perf_timer:
    def __init__(self, name='', verbose=True):
        #if name and verbose:
        #    print(name)
        self.name = name
        self.verbose = verbose
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if self.name and self.verbose:
            print('%s: elapsed time %.3f milliseconds'%(self.name, self.elapsed))
    

class ConfigArgs:
    def __init__(self, input_dict, parse=True):
        """
        Initialize the object with member variables based on input_dict.
        Set up argparse to parse and update member variables from command-line arguments.

        An easy way to create the input dict is by creating a function returning its locals, e.g.:
        def config():
            w, h = 512, 512
            return locals()

        :param input_dict: Dictionary used to initialize member variables.
        """
        # default args
        self.headless = False
        self.dropbox_folder = ''
        self.save = 1

        if not isinstance(input_dict, dict):
            raise TypeError("input_dict must be a dictionary")

        # Create member variables from the dictionary
        for key, value in input_dict.items():
            if hasattr(self, key):
                print('Overriding default attr ', key)
            setattr(self, key, value)

        # Initialize argument parser
        self.parser = argparse.ArgumentParser()
        argtype = lambda v: int if type(v)==bool else type(v)

        for key, value in self.__dict__.items():
            if callable(value) or key.startswith('_'):
                continue
            self.parser.add_argument(f"--{key}", type=argtype(value), default=None)

        if parse:
            self.parse_args()

    def parse_args(self):
        """
        Parse command-line arguments and update member variables if matching arguments are provided.
        """
        args = self.parser.parse_args()
        for key in self.__dict__.keys():
            arg_value = getattr(args, key, None)
            if arg_value is not None:
                setattr(self, key, arg_value)

    def save_json(self, filepath):
        """
        Save the object's attributes to a JSON file.

        :param filepath: Path to the JSON file where data will be saved.
        """
        params = {key:val for key, val in self.__dict__.items() if key != 'parser' and
                  not callable(val) and not key.startswith('_')}
        with open(filepath, 'w') as json_file:
            json.dump(params, json_file, indent=4)

    def load_json(self, filepath):
        """
        Load attributes from a JSON file and update the object's state.

        :param filepath: Path to the JSON file to load data from.
        """
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
        for key, value in data.items():
            setattr(self, key, value)

    def save_yaml(self, filepath):
        """
        Save the object's attributes to a YAML file.

        :param filepath: Path to the YAML file where data will be saved.
        """
        params = {key:val for key, val in self.__dict__.items() if key != 'parser' and
                  not callable(val) and not key.startswith('_')}
        with open(filepath, 'w') as yaml_file:
            yaml.dump(params, yaml_file, default_flow_style=False)

    def load_yaml(self, filepath):
        """
        Load attributes from a YAML file and update the object's state.

        :param filepath: Path to the YAML file to load data from.
        """
        with open(filepath, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        for key, value in data.items():
            setattr(self, key, value)

    def __repr__(self):
        """Provide a string representation of the object."""
        attrs = ', '.join(f"{key}={value}" for key, value in self.__dict__.items() if not callable(value) and not key.startswith('_') and key != 'parser')
        return f"ConfigArgs({attrs})"


class SaveHelper:
    def __init__(self, file, dest_dir,
                 suffix='',
                 use_wandb=False,
                 wandb_user='color-motor',
                 dropbox_folder='',
                 increment_names=True,
                 cfg={}):
        from . import fs
        save_run = True
        self.valid = False
        self.file = file
        self.dest_dir = dest_dir
        self.wandb = None
        self.dropbox_folder = dropbox_folder
        self.last_path = None
        self.collected_paths = []

        if not os.path.isdir(dest_dir):
            print("Destination dir does not exist, not saving")
            # raise ValueError
        else:
            self.valid = True
            dest_dir = os.path.abspath(dest_dir)
            self.dest_dir = dest_dir
            
            if suffix:
                suffix = '_' + suffix
            prefix = ''
            if dropbox_folder:
                # If we use a dropbox folder append a suffix to avoid conflicts
                import socket
                hostname = socket.gethostname()
                prefix = hostname.split('-')[0] + '_'

            base_name = prefix + fs.filename(file) + suffix + '_'
            
            ind = 0
            save_name = ''
            if increment_names:
                for file in fs.files_in_dir(dest_dir, '.json'):
                    if base_name in file:
                        name = fs.filename(file)
                        tokens = name.split('_')
                        try:
                            i = int(tokens[-1])
                            if i > ind:
                                ind = i
                        except ValueError:
                            pass
            self.name = base_name + '%03d'%(ind+1)
            self.save_name = os.path.join(dest_dir, self.name)
            if use_wandb:
                import wandb
                self.wandb = wandb
                wandb.init(project=fs.filename(self.file),
                           entity=wandb_user,
                           config=cfg,
                           name=self.name,
                           id=wandb.util.generate_id())

    def log(self, *args, **kwargs):
        if self.wandb is None:
            return
        self.wandb.log(*args, **kwargs)

    def log_image(self, name, img):
        if self.wandb is None:
            return
        self.wandb.log({name: self.wandb.Image(img)})

    def finish(self, filename=''):
        if filename:
            self.collected_paths.append(filename)
        if self.wandb is not None:
            self.wandb.finish()
        self.collected_to_dropbox()

    def with_ext(self, ext, suffix=''):
        if not self.valid:
            return ''
        
        if ext[0] != '.':
            ext = '.' + ext
        self.last_path = self.save_name + suffix + ext
        self.collected_paths.append(self.last_path)
        return self.last_path

    def in_dir(self, filename):
        if not self.valid:
            return ''
        path = os.path.join(self.dest_dir, filename)
        self.last_path = path
        self.collected_paths.append(self.last_path)
        return path

    def copy_file(self):
        if not self.valid:
            return
        
        import shutil
        self.last_path = self.save_name + '.py'
        self.collected_paths.append(self.last_path)
        shutil.copy(os.path.abspath(self.file), self.last_path)

    def clear_collected_paths(self):
        self.collected_paths = []

    def collected_to_dropbox(self, dbxpath='~/bin/dbxcli'):
        import subprocess
        if not self.dropbox_folder:
            print("No dropbox folder specified")
            return

        dbxcli = os.path.expanduser(dbxpath)

        print("Saving to dropbox")
        for file_path in self.collected_paths:
            if not os.path.isfile(file_path):
                print(f"Skipping: {file_path} (not a file or doesn't exist)")
                continue

            try:
                # Construct the dbxcli upload command
                dropbox_path = os.path.join(self.dropbox_folder, os.path.basename(file_path))
                command = [dbxcli, "put", file_path, dropbox_path]

                # Execute the command
                subprocess.run(command, check=True)
                print(f"Uploaded: {file_path} to {dropbox_path}")

            except subprocess.CalledProcessError as e:
                print(f"Failed to upload {file_path}: {e}")


class MultiLoss:
    def __init__(self, verbose=False, max_length=1000):
        self.items = {}
        self.losses = defaultdict(list)
        self.wrong_keys = set()
        self.verbose = verbose
        self.replace = {}
        self.max_length = max_length

    def add(self, name, fn, w):
        self.items[name] = [fn, w]

    def to_string(self):
        res = ''
        for key, v in self.losses.items():
            res += key + ' '
            res += ' '
            res += str(v[-1])
            res += '\n'
        return res

    def has_loss(self, key):
        if not key in self.items:
            return False
        if self.items[key][1] > 0.0:
            return True
        return False

    def update_weight(self, key, w):
        if not key in self.items:
            return
        self.items[key][1] = w

    def print(self):
        print(self.to_string())

    def get_weights(self):
        return {key: item[1] for key, item in self.items.items()}

    def anneal_weights(self, **kwargs):
        for key, w in kwargs.items():
            self.replace[key] = self.items[key][1]*w

    def replace_weights(self, **kwargs):
        ''' Temporarily replaces weights given the keys as arguments'''
        self.replace.update(kwargs)

    def set_weights(self, **kwargs):
        for key, w in kwargs.items():
            self.items[key][1] = w

    def plot(self):
        for i, (key, kloss) in enumerate(self.losses.items()):
            if key=='total' or not self.has_loss(key):
                # There is a bug in 'total'
                continue
            plt.plot(kloss, label='%s:%.4f'%(key, kloss[-1]))
        plt.legend()

    def __call__(self, **kwargs):
        loss = torch.tensor(0.0, device=config.device)
        for key, args in kwargs.items():
            if key not in self.items:
                if not key in self.wrong_keys:
                    # Print once to avoid clutter
                    print(key, 'not included in losses')
                    self.wrong_keys.add(key)
                continue
            fn, w = self.items[key]
            if key in self.replace:
               w = self.replace.pop(key)
            if w > 0:
                with perf_timer(key, verbose=self.verbose):
                    l = fn(*args)*w
                #print(key, 'dtype', l.dtype)
            else:
                l = 0
            if np.isscalar(l):
                self.losses[key].append(float(l))
            else:
                self.losses[key].append(float(l.detach().cpu())) #float(l.item()))
            if len(self.losses[key]) > self.max_length:
                self.losses[key] = self.losses[key][-self.max_length:]
            try:
                loss += l
            except RuntimeError as e:
                print(e)

            self.losses['total'].append(float(loss))
        return loss


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.cuda.manual_seed(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def step_lr_scheduler(optimizer, steps, num_steps):
    steps = sorted(steps, key=lambda x: x[0])
    def lr_lambda(current_step):
        for step_percent, lr_percent in steps[::-1]:
            if current_step >= num_steps * step_percent:
                print("Setting lr", lr_percent)
                return lr_percent
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    return scheduler


def step_linear_lr_scheduler(optimizer, step_percent, min_scale, num_steps):

    thresh = int(step_percent*num_steps)
    def lr_lambda(current_step):
        if current_step >= thresh:
            scale = 1.0 - (1-min_scale)*((current_step-thresh)/(num_steps - thresh))
            print('lr scale', scale)
            return scale
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    return scheduler


def cosine_decay(value):
    return 0.5 * (1 + np.cos(np.pi * value))


def step_cosine_lr_scheduler(optimizer, step_percent, min_scale, num_steps):
    thresh = int(step_percent*num_steps)
    def lr_lambda(current_step):
        current_step = min(current_step, num_steps)
        if current_step >= thresh:
            t = (current_step-thresh)/(num_steps - thresh)
            scale =  min_scale + (1-min_scale)*cosine_decay(t)
            # print('lr scale', scale)
            return scale

        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
    return scheduler




class SafeWithTupleLoader(yaml.SafeLoader):
    pass

# Handle !!python/tuple
def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))

# Fallback: Ignore unknown python/object/apply tags
def ignore_python_object(loader, tag_suffix, node):
    return None

SafeWithTupleLoader.add_constructor('tag:yaml.org,2002:python/tuple', construct_python_tuple)
SafeWithTupleLoader.add_multi_constructor('tag:yaml.org,2002:python/object/', ignore_python_object)


def extract_weights_from_yaml(f, reference='mse_w'):
    '''Retrieve weights (all entries with `_w`) from script parameters.'''
    with open(f, 'r') as f:
        data = yaml.load(f, Loader=SafeWithTupleLoader)
        #data = yaml.safe_load(f)

    weights = {k: v for k, v in data.items()
               if isinstance(k, str) and k.endswith('_w') and isinstance(v, (int, float))}
    return normalize_weights(weights, reference)


def normalize_weights(weights, reference='mse_w'):
    ''' Normalize weight values wrt a given reference weight'''
    if reference not in weights:
        raise ValueError(f"Reference weight '{reference}' not found in extracted weights: {list(weights.keys())}")

    ref_value = weights[reference]
    if ref_value == 0:
        raise ValueError(f"Reference weight '{reference}' has value 0, cannot normalize.")

    normalized_weights = {k: v / ref_value for k, v in weights.items()}
    return normalized_weights


def custom_excepthook(type, value, traceback):
    print("Exception occurred:", value)
    pdb.post_mortem(traceback)

    
def break_on_exception():
    sys.excepthook = custom_excepthook
