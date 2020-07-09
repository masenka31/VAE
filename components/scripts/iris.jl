"""
Activate the project!

using DrWatson
@quickactivate("components")
"""

using Plots
using Distributions
using Flux
using LinearAlgebra
using IPMeasures
using CSV
using DataFrames
# using UMAP

# load and prepare data
include(scriptsdir("funkce.jl"))
iris = DataFrame(CSV.File(datadir("datasets/iris.csv")))
X = (iris[:,2:5] |> Array)' |> Array
ndat = size(X,2)
setosa, versicolor, virginica = groupby(iris,:Species)
true_labels = vcat(zeros(50),ones(50),ones(50) .* 2)

# UMAP embedding
@time emb = umap(X,2,n_neighbors=15)
scatter(emb[1,:],emb[2,:],zcolor=true_labels)

# training data
dataT = [X[:,i] for i in 1:ndat]
data_train = zip(dataT,)

# model definition
nx = 4
nz = 3
nh = 8;

# loss function
function loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logs[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end

comp = 10
A = [Dense(nx, nh, swish) for i in 1:comp];
μ = [Dense(nh, nz) for i in 1:comp];
logs = [Dense(nh, nz) for i in 1:comp];
f = [Chain(Dense(nz, nh, swish), Dense(nh, nx)) for i in 1:comp];
opt = ADAM(0.01);

ps = Flux.params(A, μ, logs, f);
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))

# hyperparameters of the model
α = [1 / comp for i in 1:comp] # no prior information
s = 0.0005
b = 1/s

k = 1
@time for i in k:k+9
    global LI = lik_n(dataT)                # count the l_ik matrix
    global α = sum(LI, dims = 1) / ndat     # count α
    Flux.train!(loss, ps, zip(dataT,), opt)
    if isnan(loss(dataT[1]))                # break from loop if diverged to NaN
        println("NaN")
        break
    end
    if i%10 == 0                            # print ongoing results once in every 10 epochs
        L = loss.(dataT)
        prumer = round(sum(L) / ndat, digits = 4)
        max = round(maximum(L), digits = 4)
        n_vec = sum(round.(LI),dims=1)
        println("Epoch $i: \nn=$(round.(n_vec)) \navg=$prumer, max=$max")
    end
end
k = k + 10

labels = create_labels(LI)
scatter(emb[1,:],emb[2,:],zcolor=labels)

gen = generate_new(f,200,α)
emb_gen = umap(gen,2)
scatter(emb_gen[1,:],emb_gen[2,:],zcolor=labels)

scatter(X[1,:],X[2,:]);
scatter!(gen[1,:],gen[2,:])

scatter(X[1,:],X[3,:]);
scatter!(gen[1,:],gen[3,:])

scatter(X[1,:],X[4,:]);
scatter!(gen[1,:],gen[4,:])

scatter(X[3,:],X[2,:]);
scatter!(gen[3,:],gen[2,:])

scatter(X[4,:],X[2,:]);
scatter!(gen[4,:],gen[2,:])

scatter(X[3,:],X[4,:]);
scatter!(gen[3,:],gen[4,:])