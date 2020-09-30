include(scriptsdir("run_me.jl"))

exp_pars = Dict(
    "run_id" => [1,2,3,4],
    "s"   => [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001],
    "nh"  => [2,4,6]
)

for j in 1:size(dict_list(exp_pars),1)
    par = dict_list(exp_pars)[j]
    run_exp(par,dt,200) 
end