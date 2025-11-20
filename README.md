# Calligraph
Code for the paper:
### Neural Image Abstraction Using Long Smoothing B-Splines
*Daniel Berio, Michael Stroh, Sylvain Calinon, Frederic Fol Leymarie, Oliver Deussen, Ariel Shamir*

The system allows optimization of B-splines using a geometric smoothing cost on high positional derivatives withing a differentiable vector graphics pipeline. 

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

The repository also contains code for the paper:
### Image-Driven Robot Drawing with Rapid Lognormal Movements
*Daniel Berio, Guillaume Clivaz, Michael Stroh, Oliver Deussen, Sylvain Calinon, Réjean Plamondon, Frederic Fol Leymarie*

This paper follows a similar approach to enable minimum-time smoothing of trajectories described using the Sigma-lognormal model of handwriting movements (file [dslm.py](./calligraph.dslm.py) and demos including "slm" in the filename).

If you use this specific part of the code for academic purposes, please cite:

``` bibtex

@inproceedings{Berio25ROMAN,
  author = {Berio, D. and Clivaz, G. and Stroh, M. and Deussen, O. and Plamondon, R. and Calinon, S. and Leymarie, F. F.},
  booktitle = {Proc.{{IEEE}} Intl Symp.on Robot and Human Interactive Communication ({{Ro-Man}})},
  title = {Image-Driven Robot Drawing with Rapid Lognormal Movements},
  year = {2025}
}
```

## Licence
- The code/software in this repository is licensed under the *GNU GPLv3* (see [LICENCE](./LICENCE)). 
- *Artistic/Creative Use (Optional Attribution)*: This system is also rooted in an artistic research and practice component. If you use it for artistic or creative work, please consider mentioning/crediting Daniel Berio (@colormotor).

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
-   Install NumPy, SciPy, matplotlib, OpenCV (using mamba as above, or pip)
-   Install [torch/torchvision](https://pytorch.org/get-started/locally/)
    following your system specs
-   Install DiffVg from the [colormotor branch](https://github.com/colormotor/diffvg) (has thick strokes fix):
    -   clone the repo: `git clone https://github.com/colormotor/diffvg.git`
    -   From the repo directory do:
        -   `git submodule update --init --recursive` and then
        -   `python setup.py install`
-   Install remaining deps with pip:
    - `pip install accelerate transformers diffusers ortools open-clip-torch pyclipper freetype-py svgpathtools`


## Install locally
Finally, install locally from the repo directory with

    pip install -e .


# Examples

Examples are located in the [examples](./examples) directory. By default the outputs are saved in a `generated` subdirectory at the same level as the example. If this direcory does not exist the outut will not be saved. Create the directory or rename the parameter if you wish to save the results.

## Parameter settings
In each example, configuration parameters are set by adding local variables to a `params()` function. These are automatically converted to command-line arguments that can be set when executing a script. E.g. 

``` 
python demo_area_fill_01.py --smoothing_w=10
```

Will the set the value of the corresponding variable declared inside the `params()` function.




