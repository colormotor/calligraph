# Calligraph
Code for the paper:
### Neural Image Abstraction Using Long Smoothing B-Splines

If you use this code for academic purposes please cite:


``` bibtex
@article{NeurSplines-25,
	title = {Neural Image Abstraction using Long Smoothing B-Splines},
	author = {Daniel Berio and Michael Stroh and Sylvain Calinon and Frederic Fol Leymarie and Oliver Deussen and Ariel Shamir },
	journal = {ACM Transactions on Graphics (SIGGRAPH Asia 2025 Conference Proceedings)},
	year = {2025},
	volume = {44},
	Number = {6},
	pages = {Accepted},
}
```

## Conda (recommended)

The ideal way to get this working is installing the conda/mamba package manager through miniforge. On Mac/Linux, from a terminal do

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

It is recommended to create a new environment to install the dependencies, which can be done with
 
    mamba create -n calligraph python=3.10
    
You can replace `calligraph` with any name of your preference. Activate the env:

    mamba activate calligraph

Then proceed with the following dependencies. If using conda/mamba you may want to install these beforehand:

    mamba install numpy scipy matplotlib opencv scikit-image

making sure your environment is active.
    
## Dependencies
-   Install NumPy, SciPy, matplotlib, OpenCV (using mamba as above or pip)
-   Install [torch/torchvision](https://pytorch.org/get-started/locally/)
    following your system specs
-   Install DiffVg from the [colormotor branch](https://github.com/colormotor/diffvg) (has thick strokes fix):
    -   clone the repo: `git clone https://github.com/colormotor/diffvg.git`
    -   From the repo directory do:
        -   `git submodule update --init --recursive` and then
        -   `python setup.py install`
-   Install remaining deps with pip:
    - `pip install accelerate transformers diffusers ortools open-clip-torch`


## Install locally

Finally, install locally from the repo directory with

    pip install -e .


# Examples

Examples are located in the [examples](./examples) directory. By default the outputs are saved in an outputs directory. If this direcory does not exist the outut will not be saved. that will be automatically created. If this directory does not exist, no output is saved. In each example, configuration parameters are set by adding local variables to a `params()` function. These are automatically converted to command-line arguments that can be set when executing a script. 


