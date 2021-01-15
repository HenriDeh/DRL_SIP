# Instructions for experiment reproducibility
of the paper "_A Deep Reinforcement Learning approach to the Stochastic Inventory Problem_" (full citation when accepted).
## Installation and dependencies

1. Install Julia. The official version of this paper is 1.5.1 but any older version should work fine (if not better).
2. Clone this project to your local computer to any path you desire.
3. Start Julia and change the working directory to that of the DRL_SIP project. You can do so with the `cd("path/of/DRL_SIP")` julia function. Use `pwd()` to check your working directory.
4. Download all dependencies. Julia's package manager will do this for you. Simply execute the following functions:
    1. `using Pkg`
    2. `Pkg.activate()`
    3. `Pkg.instantiate()`  

    You can also copy paste `using Pkg; Pkg.activate(); Pkg.instantiate()` to do everything at once. This operation is only necessary the first time you use the project.

**A note on CUDA and GPU acceleration**  
You may encounter difficulties with the installation of `CUDA.jl` artifacts, if so, refer to the [documentation](https://juliagpu.github.io/CUDA.jl/stable/installation/overview/#InstallationOverview). The project can in principle work without GPU acceleration but will be extremely slow. It is unfortunately not compatible with non NVidia Graphics Cards.

## Reproducing the experiments
To reproduce all experiments described in the paper, simply include the script with `include("scripts/main_experiments/all_experiments.jl")`. Warning, the experiments are extensive and take weeks to finish. You can reproduce a subset of the experiments by commenting the lines (with `#`) of the experiments you are not interested in. This operation will overwrite the output data files in the `data/main_experiments` folder. 

## Making custom experiments.
The `experiment()` function allows for user customization of the environment parameters, algorithm hyperparameters, output paths, and more. Refer to [our documentation]() to make custom experiments on the SL-SIP.

## More documentation on this project
Visit our documentation page for detailed explanations about the code in this project.
