"""
Initialize DrWatson project.

using DrWatson
@quickactivate("components")
"""

# include predefined functions
include(scriptsdir("funkce.jl"))
# load data
include(scriptsdir("init_pulsars.jl"))

# initialize network
nx = 8
nz = 8
nh = 30
comp = 5
α = [1 / comp for i in 1:comp] # no prior information
s = 0.5
b = 1 / s
η = 0.0001 
opt = ADAM(η)
ep = 300

A = [Dense(nx, nh, sigmoid) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logs = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, sigmoid), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logs, f);
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))

function loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logs[i](A[i](x))) - log(α[i] + 10^(-8))), 1:comp)
    dot(li, o)
end

# initial computation of labels and α
global LI = lik_n(dataN_train)                         # count the l_ik matrix
global α = sum(LI, dims = 1) / train_size     # count α
x = dataN[1] 
l_prev = loss(x)

# training procedure
@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global LI = lik_n(dataN)                            # count the l_ik matrix
    global α = sum(LI, dims = 1) / normal_count       # count α
    global l = loss(x)
    if isnan(l)
        println("loss diverged into NaN, training terminated...")
        break
    end
    LN = loss.(dataN)
    LA = loss.(dataA)
    avg_norm = mean(LN)
    avg_an = mean(LA)
    if i%5 == 0
        L = loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr, tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s,α,comp)
        safesave(datadir("pulsars-ALL_DATA", savename("run-hard-6", final_model, "bson")), final_model)
        println("AUC: $(round(auc1,digits=4))")
    end
    n_vec = sum(round.(LI),dims=1)
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an)), n: $n_vec")
end

# save the final model after specified number of epochs
# final_model = @dict(ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s,c,n_sample)
# safesave(datadir("WAE_final", savename("final_wae_gaussian", final_model, "bson")), final_model)

printstyled("Current experiment completed.\n", bold = true, color = :cyan)

