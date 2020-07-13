"""
Initialize DrWatson project.

using DrWatson
@quickactivate("components")
"""

using Plots
using Flux
using Distributions
using LinearAlgebra
# using LaTeXStrings
# pgfplotsx()

include(scriptsdir("funkce.jl"))
include(scriptsdir("init_kdd.jl"))

# initialize network
nx = 22
nz = 10
nh = 30
comp = 10
α = [1 / comp for i in 1:comp] # no prior information
s = 0.03
b = 1 / s
opt = ADAM(0.01)

A = [Dense(nx, nh, σ) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logs = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, σ), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logs, f);
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))


function loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logs[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end

# training procedure
l_prev = loss(dataT[1])
@time for i in 1:ep
    global LI = lik_n(dataT)                # count the l_ik matrix
    global α = sum(LI, dims = 1) / ndat     # count α
    Flux.train!(loss, ps, data_train, opt)
    global l = loss(dataT[1])
    if isnan(l)
        "loss diverged into NaN, training terminated..."
        break
    end
    if i%20 == 0
        L = loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr, tpr)
        global FPR_end = fpr
        global TPR_end = tpr
        global auc_end = auc1
        final_model = @dict(i,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s)
        safesave(datadir("KDD", savename("emvae", final_model, "bson")), final_model)
    end
    n_vec = sum(round.(LI),dims=1)
    println("Epoch: $i, loss: $l, n: $n_vec")
end

println("Results: \n maximum AUC = $auc_end")

# save the final model after specified number of epochs
# final_model = @dict(ep,A,μ,logs,f,opt,FPR_end,TPR_end,auc_end,s,c,n_sample)
# safesave(datadir("WAE_final", savename("final_wae_gaussian", final_model, "bson")), final_model)

printstyled("Current experiment completed.\n", bold = true, color = :cyan)