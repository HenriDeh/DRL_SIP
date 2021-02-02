using DrWatson
@quickactivate "DRL_SIP"
using InventoryModels, CSV, DataFrames, ProgressMeter, BSON

CSV.write(datadir("instances_solved.csv"), DataFrame(Experiment = [], parameters_ID = [], instance_ID = [], opt = []))
forecasts = CSV.File("data/instances.csv") |> DataFrame!
#backlog
setups = [80,360,1280]
backorders = [5,10]
CVs = [0.2,0.4]
production = 0
holding = 1
H = 52

i = 0
for setup in setups, backorder in backorders, CV in CVs
    global i += 1
    println("Presolving Dataset Backlog ($setup, $backorder, $CV)")
    df = DataFrame(Experiment = [], parameters_ID = [], instance_ID = [], opt = [])
    @showprogress for k in 1:nrow(forecasts)
        forecast = eval(Meta.parse(forecasts[k,:].forecast))
        #build test environment
        reseter = CVNormal.(forecast, CV)
        sup = Supplier(fixed_linear_cost(setup,production))
        pro = ProductInventory(linear_holding_cost(holding), sup, Normal(0, 0))
        ma = Market(linear_stockout_cost(backorder), pro, CVNormal, true, reseter, expected_reward = false, horizon = H)
        test_envi = InventoryProblem([sup, pro, ma])
        #evaluate policies
        instance = Instance(test_envi)
        Scarf.backward_SDP(instance, 0.1)
        opt_policy_value = test_Scarf_policy(test_envi, instance.S, instance.s)
        push!(df, ["backlog", i, k, opt_policy_value])
    end
    CSV.write(datadir("instances_solved.csv"), df, append = true)
end

leadtimes = [0,1,2,4,8]
CV = 0.4
stockout = 10

i = 0
for setup in setups, leadtime in leadtimes
    global i += 1
    println("Presolving Dataset Lead Time ($setup, $leadtime)")
    df = DataFrame(Experiment = [], parameters_ID = [], instance_ID = [], opt = [])
    @showprogress for k in 1:nrow(forecasts)
        forecast = eval(Meta.parse(forecasts[k,:].forecast))
        #build test environment
        reseter = CVNormal.(forecast, CV)
        sup = Supplier(fixed_linear_cost(setup,production), zeros(leadtime))
        pro = ProductInventory(linear_holding_cost(holding), sup, Normal(0, 0))
        ma = Market(linear_stockout_cost(stockout), pro, CVNormal, true, reseter, expected_reward = false, horizon = H)
        test_envi = InventoryProblem([sup, pro, ma])
        #evaluate policies
        instance = Instance(test_envi)
        Scarf.backward_SDP(instance, 0.1)
        opt_policy_value = test_Scarf_policy(test_envi, instance.S, instance.s)
        push!(df, ["leadtime", i, k, opt_policy_value])
    end
    CSV.write(datadir("instances_solved.csv"), df, append = true)
end

CV = 0.4
backorders = [10,25,50,100]
i = 0
for setup in setups, backorder in backorders
    global i += 1
    println("Presolving Dataset Lost Sales ($setup, $backorder")
    df = DataFrame(Experiment = [], parameters_ID = [], instance_ID = [], opt = [])
    @showprogress for k in 1:nrow(forecasts)
        forecast = eval(Meta.parse(forecasts[k,:].forecast))
        #build test environment
        reseter = CVNormal.(forecast, CV)
        sup = Supplier(fixed_linear_cost(setup,production))
        pro = ProductInventory(linear_holding_cost(holding), sup, Normal(0, 0))
        ma = Market(linear_stockout_cost(backorder), pro, CVNormal, false, reseter, expected_reward = false, horizon = H)
        test_envi = InventoryProblem([sup, pro, ma])
        #evaluate policies
        instance = Instance(test_envi)
        Scarf.backward_SDP(instance, 0.1)
        opt_policy_value = test_Scarf_policy(test_envi, instance.S, instance.s)
        push!(df, ["lostsales", i, k, opt_policy_value])
    end
    CSV.write(datadir("instances_solved.csv"), df, append = true)
end