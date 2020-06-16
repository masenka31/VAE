# skript for running the experiment
all_parameters = Dict(
    "run_id" => 4,#[1,2,3,4],
    "nx" => 22,
    "nz" => [2,6,10],
    "nh" => [10,30],
    "η"  => [0.001],
    "opts" => [ADAM, RMSProp],
    "activ" => [σ, swish],
    "s" => [1, 0.1, 10]        
)

for j in 1:size(dict_list(all_parameters),1)
    par = dict_list(all_parameters)[j]
    run_exp(par,dataT,50) 
end


bl = [:A,:FPR,:TPR,:auc_progress,:μ,:logs,:f]
res = collect_results(datadir("results"),black_list=bl)

# data analysis?

filtered = filter(:η => x -> x == 0.01,dropmissing(res,:auc_max))
filtered = filter([:η, :opts] => (x, y) -> x == 0.01 && y == ADAM ,dropmissing(res,:auc_max))
filter(:auc_max => x -> x > 0.99,dropmissing(res,:auc_max))
filter(:auc_max => x -> x > 0.99,dropmissing(filtered,:auc_max))

avg = evaluate_averages(res,:nh,:auc_max)

function evaluate_averages(df,gb_symb::Symbol,mean_symb::Symbol)
    by_symb = groupby(df,gb_symb)
    ln = length(by_symb)
    avr = hcat(keys(by_symb),ones(ln))
    for i in 1:ln
        avr[i,2] = mean(skipmissing(by_symb[i][!,mean_symb]))
    end
    ret = avr |> DataFrame
    sort!(ret,:x2,rev=true)
    return ret
end

avg_s = evaluate_averages(res,:s,:auc_max)
avg_nh = evaluate_averages(res,:nh,:auc_max)
avg_opt = evaluate_averages(res,:opts,:auc_max)


# finds entries with auc_max> 0.99  
filter(:auc_max => x -> x > 0.99,dropmissing(res,:auc_max))