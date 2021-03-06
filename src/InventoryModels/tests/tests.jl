using Test, Revise
using InventoryModels

@testset "Supplier" begin
	sup = Supplier(fixed_linear_cost(100,1), Float64[])
	@test sup.lead_time == 0 == length(sup.orders) == length(sup.reset_orders) == length(sup.test_reset_orders)
	@test observe(sup) == Float64[]
	reset!(sup)
	@test observe(sup) == Float64[]
	test_reset!(sup)
	@test observe(sup) == Float64[]
	@test sup(10) == 110
	@test observe(sup) == [10]
	test_reset!(sup)
	@test sup(-10) == 0
	sup = Supplier(fixed_linear_cost(100,1), zeros(2))
	@test sup.lead_time == 2 == length(sup.orders) == length(sup.reset_orders) == length(sup.test_reset_orders)
	@test observe(sup) == [0,0]
	sup = Supplier(fixed_linear_cost(100,1), fill(Uniform(0,10),2), test_reset_orders=fill(2,2))
	@test sup.lead_time == 2 == length(sup.orders) == length(sup.reset_orders) == length(sup.test_reset_orders)
	test_reset!(sup)
	@test observe(sup) == [2,2]
	@test sup(10) == 110
	@test observe(sup) == [2,2,10]
	reset!(sup)
	@test observe(sup) != [2,2] != [2,2,10]
end

@testset "Inventory" begin
	sup = Supplier(fixed_linear_cost(100,1), fill(2,2))
	inven = Inventory(linear_cost(4), sup, 3, 0, test_reset_level =0)
	@test observe(inven) == 3
	@test sup(10) == 110
	@test inven(0) == 5*4 + 12*4
	sup = Supplier(fixed_linear_cost(100,1), fill(2,2))
	inven = Inventory(linear_cost(4), sup, Uniform(0,20), 0, test_reset_level =3)
	test_reset!(inven)
	@test observe(inven) == 3
	@test sup(10) == 110
	@test inven(0) == 5*4 + 12*4
end

@testset "ProductInventory" begin
	sup = Supplier(fixed_linear_cost(100,0))
	inven = ProductInventory(linear_holding_cost(1), sup, 0.0)
	@test observe(inven) == 0
	@test sup(10) == 100
	@test inven(0) == 10
	inven = ProductInventory(linear_holding_cost(1), sup, Uniform(-10,20), test_reset_level = 0.0)
	@test observe(inven) != 0
	test_reset!(inven)
	@test observe(inven) == 0
	@test sup(10) == 100
	@test inven(10) == 0
	@test inven.holding_cost(inven.level) == 0
	reset!(inven)
	@test observe(inven) != 0
end

@testset "Martket" begin
	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(expected_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv= 0.25
	ma = Market(expected_stockout_cost(10), pro, Normal, true, Normal.(forecasts, cv.*forecasts), expected_reward = true)
	@test observe(ma) == [20,5,40,10,60,15,40,10]
	@test observation_size(ma) == length(observe(ma))
	ma = Market(expected_stockout_cost(10), pro, CVNormal, true, CVNormal.(forecasts, cv))
	@test observe(ma) == [20,40,60,40]
	@test observation_size(ma) == length(observe(ma))
	reset!(ma)
	@test observe(ma) == [20,40,60,40]
	ma.t = 2
	@test observe(ma) == [40,60,40,0]
	ma.t = 3
	@test observe(ma) == [60,40,0,0]
	ma.t = 4
	@test observe(ma) == [40,0,0,0]
	ma.t = 5
	@test observe(ma) == [0,0,0,0]
	test_reset!(ma)
	@test observe(ma) == [20,40,60,40]
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(expected_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv),expected_reward = true)
	test_reset!(ma)
	@test observe(ma) == [20,40,60,40]
	reset!(ma)
	@test observe(ma) != [20,40,60,40]

	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(expected_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.25
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(expected_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv), expected_reward = true)
	test_reset!(ma)
	sup(10)
	@test ma() ≈ 100.46663620521514 ≈ expected_stockout_cost(10)(10, Normal(20,5)) + expected_holding_cost(1)(10, Normal(20,5))

	sup = Supplier(fixed_linear_cost(100,0), zeros(1))
	pro = ProductInventory(expected_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.25
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(expected_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv),expected_reward = true)
	test_reset!(ma)
	sup(10)
	@test ma() ≈ expected_stockout_cost(10)(0, Normal(20,5)) + expected_holding_cost(1)(0, Normal(20,5))

	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(linear_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.0
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(linear_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv), expected_reward = false)
	test_reset!(ma)
	sup(10)
	@test ma() == 100
	sup(60)
	@test ma() == 10

	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(linear_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.0
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(linear_stockout_cost(10), pro, CVNormal, false, reseter, test_reset_forecasts = CVNormal.(forecasts, cv), expected_reward = false)
	test_reset!(ma)
	sup(10)
	@test ma() == 100
	sup(60)
	@test ma() == 20
end

@testset "InventoryProblem" begin
	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(expected_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.25
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(expected_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv),expected_reward = true)
	envi = InventoryProblem([sup, pro, ma])
	test_reset!(envi)
	@test observe(envi) == [0,20,40,60,40]
	reset!(envi)
	@test observe(envi) != [0,20,40,60,40]
	@test observation_size(envi) == 5
	@test action_size(envi) == 1
	test_reset!(envi)
	@test envi([10]) ≈ -100.46663620521514 - 100
	envi(0)
	envi(100)
	envi(0)
	@test isdone(envi)
end

@testset "Scarf" begin
	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(linear_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.25
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(linear_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv),expected_reward = false)
	envi = InventoryProblem([sup, pro, ma])
	test_reset!(envi)
	instance = Instance(envi)
	Scarf.backward_SDP(instance)
	@test instance.s == [14, 29, 58, 28]
	@test instance.S == [70, 141, 114, 53]
	@test -363 < test_Scarf_policy(envi, instance.S, instance.s) < -362

	sup = Supplier(fixed_linear_cost(100,0), zeros(1))
	pro = ProductInventory(linear_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.25
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(linear_stockout_cost(10), pro, CVNormal, true, reseter, test_reset_forecasts = CVNormal.(forecasts, cv),expected_reward = false)
	envi = InventoryProblem([sup, pro, ma])
	test_reset!(envi)
	instance = Instance(envi)
	Scarf.backward_SDP(instance)
	@test instance.s == [49, 102, 90, -Inf]
	@test instance.S == [172, 159, 124, -Inf]

	sup = Supplier(fixed_linear_cost(100,0))
	pro = ProductInventory(linear_holding_cost(1), sup, 0.0)
	forecasts = [20,40,60,40]
	cv = 0.25
	reseter = repeat([(Uniform(0,20), Normal(cv,0))],4)
	ma = Market(linear_stockout_cost(10), pro, CVNormal, false, reseter, test_reset_forecasts = CVNormal.(forecasts, cv),expected_reward = false)
	envi = InventoryProblem([sup, pro, ma])
	test_reset!(envi)
	instance = Instance(envi)
	Scarf.backward_SDP(instance)
	@test instance.s == [14, 29, 58, 28]
	@test instance.S == [70, 141, 114, 53]
end
