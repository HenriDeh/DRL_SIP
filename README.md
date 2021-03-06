# Instructions for experiment reproducibility
of the paper "_A Deep Reinforcement Learning approach to the Stochastic Inventory Problem_" (full citation when accepted).
## Installation and dependencies

1. Install Julia. The official version of this paper is 1.5.1 but any version older than 1.5 should work fine (if not better).
2. Clone this project to your local computer to any path you desire.
3. Start Julia and change the working directory to that of the DRL_SIP project. You can do so with the `cd("path/of/DRL_SIP")` julia function. Use `pwd()` to check your working directory.
4. Download all dependencies. Julia's package manager will do this for you. Simply execute the following functions:
    1. `using Pkg`
    2. `Pkg.activate()`
    3. `Pkg.instantiate()`  

    You can also copy paste `using Pkg; Pkg.activate(); Pkg.instantiate()` to do everything at once. The third (iii.) operation is only necessary the first time you use the project.

**A note on CUDA and GPU acceleration**  
You may encounter difficulties with the installation of `CUDA.jl` artifacts, if so, refer to the [documentation](https://juliagpu.github.io/CUDA.jl/stable/installation/overview/#InstallationOverview). The project can in principle work without GPU acceleration but will be extremely slow. It is unfortunately not compatible with non NVidia Graphics Cards.

## Reproducing the experiments
To reproduce all experiments described in the paper, simply include the script in Julia with `include("scripts/main_experiments/all_experiments.jl")`. Warning, the experiments are extensive and take weeks to finish. You can reproduce a subset of the experiments by commenting the lines (with `#`) of the experiments you are not interested in. This operation will overwrite the output data files in the `data/main_experiments` folder.

## Data files
All data files are available in the data folder.  
`instances.csv` contains the 500 generated forecasts for the dataset, they were generated with the `script/instances.jl` script.  
`instances_solved` contain the optimal value of all 500*(12+15+12)*20= 19500 instances solved for this paper. The file was generated with the script `script/main_experiments/scarf_presolving.jl`  
The output data of the experiments are available in the folder `data/exp_raw/main_experiments/`. The name of each file allows to recognize from which experiment it was generated: `version-h-e-a-t(_returns/details/bag).csv`.

* version is one of "backlog","leadtime","lostsales";
* h is "hybrid" if the hybrid component was activated, "continuous" if not;
* e is "expected" if the reward returned by the environment is the computed expectation, "sample" if it's a randomized reward;
* a is "Kannealed" if the 75000 iterations annealing was activated, "Kfixed" if not;
* t is "twin" if the twin critic is used, "no_twin" if not.

Each experiment generates four csv files using the above naming. One has no appendage, it contains the mean gap of each agent over the 500 instances. One is apprended with "\_details", it contains the gap of each agent for each of the 500 instances. One is appended with "\_returns", it contains the gap with respect to the test instance with stationnary demand, computed every 3000 iterations during training. One is appended with "\_bag", it contains the performance measures of the ensemble agents.


## Making custom experiments.
The `experiment()` function allows for user customization of the environment parameters, algorithm hyperparameters, output paths, and more. Here is the documentation of this function.
```
	experiment(variant::String = "backlog"; iterations = 300000, 
		twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, 
		expected_reward::Bool = true, N = 20, 
		critic_lr = 1f-4, actor_lr = critic_lr/8, actor_clip = 1f-6, discount = 0.99, 
		softsync_rate = 0.001, batchsize = 128, replaysize = 2^15, width = 64, 
		epsilon = 0.005, folder = "main_experiments", overwrite = true, kwargs...)
```
Run the experiments of the specified `variant` with the specified components.  
* `variant = "backlog"` change to "leadtime" or "lostsales" to run the respective experiments. This is an ordinary argument, not a keyword argument. 

Use the keyword arguments to customize the experiments:
* `iterations = 300000` number of training iterations per agent.
* `twin = true` set to `false` to disable the twin critic.
* `annealing = 75000` set to `0` to disable the linear annealing of ``K``, set to any value to change the rate of annealing. 
* `hybrid = true` set to false to disable the discrete check.
* `expected_reward = true` set false to make the environment return a "normal" sampled reward instead of the expectation.
* `N = 20` set the number of agents trained for each environment parameter combination.
* `critic_lr = 1f-4` set the learning rate of the critic ``α_C``.
* `actor_lr = critic_lr/8` set the learning rate of the actor ``α_A``.
* `actor_clip = 1f-6` set the maximum norm of the gradient for clipping.
* `discount = 0.99` set the discount factor. 
* `softsync_rate = 0.001` set the synchronization rate ``β``.
* `batchsize = 128` set the batch size ``s_B``.
* `replaysize = 2^15` set the experience replay size ``s_R``.
* `width = 64` set the number of neurons per hidden layer`.
* `epsilon = 0.005` set the probability `ϵ` to use the exploration strategy.
* `folder = "main_experiments"` change the output folder where the experiments' CSV are recorded.
* `overwrite = true` set to false to append instead of erasing the old experiement CSV.
* `experimentname` give a user-defined experiment name for the CSV files, automatically generated if not provided.
* Use kwargs to define custom _environment parameter_ sets. Example:
    ```
    experiment("backlog",
    stockouts = [10],
    CVs = [0.4],
    order_costs = [1280],
    leadtimes = [0])
    ```
    will only experiment on one instance. Unspecified environment parameter sets will be defaulted to the respective values of the `variant` argument.
    The accepted keywords are the four above.
