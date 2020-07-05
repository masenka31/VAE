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

res = collect_results(datadir("results"))


L = 20
function rec_prob(x)
    HID = A(x);
    zsample = hcat([rand(MvNormal(μ(HID),logs(HID))) for i in 1:L]...)
    E = hcat([f(zsample[:,i]) for i in 1:L]...)
    1/L*sum([pdf(MvNormal(E[:,i],s*I),x) for i in 1:L])
end

function rec_prob(x)
    HID = A(x);
    zsample = hcat([z(μ(HID), logs(HID)) for i in 1:L]...)
    E = hcat([f(zsample[:,i]) for i in 1:L]...)
    1/L*sum([pdf(MvNormal(E[:,i],s*I),x) for i in 1:L])
end