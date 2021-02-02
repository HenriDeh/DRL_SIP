# DRL_SIP Documentation

```julia
experiment(variant::String = "backlog"; iterations = 300000, 
		twin::Bool = true, annealing::Int = 75000, hybrid::Bool = true, 
		expected_reward::Bool = true, N = 20, 
		critic_lr = 1f-4, actor_lr = critic_lr/8, actor_clip = 1f-6, discount = 0.99, 
		softsync_rate = 0.001, batchsize = 128, replaysize = 2^15, width = 64, 
		epsilon = 0.005, folder = "main_experiments", overwrite = true, kwargs...)
```
Run the experiments of the specified `variant` with the specified components.  

Use the keyword arguments to customize the experiments:
* `variant = "backlog"` change to "leadtime" or "lostsales" to run the respective experiments.
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
