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

# create data to learn
X = rand(MvNormal([0;5;5], I), 800);
Y = rand(MvNormal([-5;-3;0], I), 200);
Z = rand(MvNormal([5;-3;-7], I), 400);
dt = hcat(X, Y, Z);
scatter(dt[1,:],dt[2,:],dt[3,:],aspect_ratio=:equal)
ndat = size(dt, 2);
dataT = [dt[:,i] for i in 1:ndat];
opt = ADAM(0.01)

# initialize network
nx = 3
nz = 3
nh = 5

comp = 6
A = [Dense(nx, nh, swish) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logs = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, swish), Dense(nh, nx)) for i in 1:comp]
opt = ADAM(0.01)
ps = Flux.params(A, μ, logs, f);
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))


function loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logs[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end

# create new network
A = [Dense(nx, nh, swish) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logs = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, swish), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logs, f);
opt = ADAM(0.01)
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))

α = [1 / comp for i in 1:comp] # no prior information
s = 0.03
b = 1 / s

k = 1
@time for i in k:k+49
    global LI = lik_n(dataT)                # count the l_ik matrix
    global α = sum(LI, dims = 1) / ndat     # count α
    Flux.train!(loss, ps, zip(dataT,), opt)
    if isnan(loss(dataT[1]))
        println("NaN")
        break
    end
    if i%10 == 0
        L = loss.(dataT)
        prumer = round(sum(L) / ndat, digits = 4)
        max = round(maximum(L), digits = 4)
        n_vec = sum(round.(LI),dims=1)
        println("Epoch $i: \nn=$(round.(n_vec)) \navg=$prumer, max=$max")
    end
end
k = k + 50

labels = create_labels(LI)
scatter(dt[1,:],dt[2,:],dt[3,:],zcolor=labels,aspect_ratio=:equal,label="data classes")

gen = generate_new(f,700,α)
gen_labels = create_labels_new(gen)
scatter(dt[1,:],dt[2,:],dt[3,:],legend=:topright,label="data",aspect_ratio=:equal);
scatter!(gen[1,:],gen[2,:],gen[3,:],label="f(e)")
# savefig(plotsdir("3gaussians_data.pdf"))

scatter(X[1,:],X[2,:],label="X",size=(500,300),aspect_ratio=:equal,legend=:topright);
scatter!(Y[1,:],Y[2,:],label="Y");
scatter!(Z[1,:],Z[2,:],label="Z")
scatter!(gen[1,:],gen[2,:],label=L"f(\varepsilon)")

# with zcolor
scatter(dt[1,:],dt[2,:],legend=:topright,label="data",aspect_ratio=:equal,zcolor=labels);
scatter!(gen[1,:],gen[2,:],label="f(e)",color=:green)
# zcolor generated data
scatter(gen[1,:],gen[2,:],label="gen classes",zcolor=gen_labels,aspect_ratio=:equal)