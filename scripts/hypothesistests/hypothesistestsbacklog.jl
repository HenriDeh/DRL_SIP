using HypothesisTests, Combinatorics, CSV, DataFrames, StatsBase, Statistics, Query

tb = CSV.File("data/exp_raw/main_experiments/backlog-hybrid-expected-Kfixed-notwin.csv") |> DataFrame!
tb.ablation = fill("1_Vanilla", nrow(tb))
ta = CSV.File("data/exp_raw/main_experiments/backlog-hybrid-expected-Kannealed-notwin.csv") |> DataFrame!
ta.ablation = fill("2_No twin", nrow(ta))
td = CSV.File("data/exp_raw/main_experiments/backlog-hybrid-expected-Kfixed-twin.csv") |> DataFrame!
td.ablation = fill("3_No annealing", nrow(td))
tc = CSV.File("data/exp_raw/main_experiments/backlog-hybrid-expected-Kannealed-twin.csv") |> DataFrame!
tc.ablation = fill("4_Full", nrow(tc))
df = transform(vcat(tb,ta,td,tc), :mean_gap => ByRow(x->x>0.15) => :fail)
t = filter(r -> r.fail == 0, df)
#Analysis
#environment
begin
    testdf = DataFrame(Baseline = [], Ablation = [], gap =[], diff = [], pvalue_mean = [], fail = [], fail_diff = [], fail_pvalue = [])
    testdf2 = DataFrame(Baseline = [], Ablation = [], gap =[], diff = [], pvalue_mean = [], fail = [], fail_diff = [], fail_pvalue = [])
    nogroup = groupby(t, [:ablation])
    nogroupf = groupby(df, [:ablation])
    for (key2,subdf2) in pairs(filter(x->x.ablation[1] ∈("1_Vanilla", "3_No annealing", "2_No twin"), nogroup))
        for (key1, subdf1) in pairs(filter(x->x.ablation[1] ∈("4_Full",), nogroup))
            #Mean test
            x = subdf1.mean_gap
            y = subdf2.mean_gap
            test = UnequalVarianceTTest(x, y)
            #Fail test
            xf = nogroupf[(key1.ablation,)].fail
            yf = nogroupf[(key2.ablation,)].fail
            testf = FisherExactTest(sum(xf), sum(yf), length(xf), length(yf))
            push!(testdf, [key1.ablation[3:end], key2.ablation[3:end], mean(y), mean(y) - mean(x), pvalue(test), testf.b/testf.d, -(testf.a/testf.c - testf.b/testf.d), pvalue(testf)])
        end
    end
    transform!(testdf, names(testdf)[3:8] .=> x->round.(x, sigdigits = 2), renamecols = false)
    for (key1, subdf1) in pairs(filter(x->x.ablation[1] ∈["4_Full"], nogroup))
        for (key2,subdf2) in pairs(filter(x->x.ablation[1] ∈["1_Vanilla"], nogroup))
            #Mean test
            x = subdf1.mean_gap
            y = subdf2.mean_gap
            test = UnequalVarianceTTest(x, y)
            #Fail test
            xf = nogroupf[(key1.ablation,)].fail
            yf = nogroupf[(key2.ablation,)].fail
            testf = FisherExactTest(sum(xf), sum(yf), length(xf), length(yf))
            push!(testdf2, ["Full", "Full", round(mean(x), sigdigits = 2), "", "", round(testf.a/testf.c, sigdigits = 2), "", ""])
            #push!(testdf2, ["Vanilla", "", round(mean(y), sigdigits = 2), "", "", round(testf.b/testf.d, sigdigits = 2), "", ""])
            #push!(testdf2, [key1.ablation[3:end], key2.ablation[3:end], round.([mean(y), mean(y) - mean(x), pvalue(test), testf.b/testf.d, -(testf.a/testf.c - testf.b/testf.d)], sigdigits = 2)..., pvalue(testf)])
        end
    end
    testdf = vcat(testdf2,testdf)
end

select!(testdf, names(testdf) .=> ["Baseline", "Ablation", "Mean Gap", "Gap Diff", "Gap p-value", "Fail prob.", "Fail Diff", "Fail p-value"])

CSV.write("data/htests/backlog.csv", testdf[:,Not(:Baseline)], delim = "&", newline = "\\\\\n")


show(testdf)