all_parameters = Dict(
    "run_id" => 4, #[1,2,3,4],
    "nx" => 8,
    "nz" => [2,5,8],
    "nh" => [2,5,10,30],
    "η"  => [0.01,0.001],
    "opts" => [ADAM, RMSProp],
    "activ" => [σ, swish],
    "s" => [1, 0.1, 10]
)

for j in 1:size(dict_list(all_parameters),1)
    par = dict_list(all_parameters)[j]
    run_exp(par,dataT,50) 
end
