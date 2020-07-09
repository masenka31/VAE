using Plots
using Flux
using Distributions
using LinearAlgebra

include("C:\\Users\\masen\\OneDrive\\SCHOOL\\BP\\programming\\funkce.jl")

X = rand(MvNormal([0;0], I), 100)
Y = rand(MvNormal([15;15], I), 100)
Z = rand(MvNormal([-15;-15], I), 100)
dt = vcat(X', Y')
scatter(dt[:,1],dt[:,2])
ndat = size(dt, 1)
dataT = [dt[i,:] for i in 1:ndat]
opt = ADAM()

dataT = [dt[i,:] for i in 1:200]
ndat = 200

nx = 2
nz = 2
nh = 2

comp = 2
A = [Dense(nx, nh, swish) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logσ = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, swish), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logσ, f);
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. * logσ)
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))


function lik(x)
    HID = [A[i](x) for i in 1:comp]
    zs = [z(μ[i](HID[i]), logσ[i](HID[i])) for i in 1:comp]
    expo = map(1:comp) do i
        temp = f[i](zs[i])
        vc = -sum((x .- temp).^2)
    end
    max_exp = maximum(expo)
    lbl = [exp(expo[i] - max_exp) * α[i] for i in 1:comp] 
    soucet = sum(lbl)
    lbl = lbl ./ soucet
    return lbl
end

function lik_n(dataT)
    LI = zeros(ndat, comp)
    for i in 1:ndat
        li = lik(dataT[i])
        for j in 1:comp
            LI[i,j] = li[j]
        end
    end
    return LI
end


function rec(x, i)
    HID = A[i](x)
    zsample = z(μ[i](HID), logσ[i](HID))
    return 0.5 * b * sum((x .- f[i](zsample)).^2)
end

function loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logσ[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end

function loss_l(x,li)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logσ[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end
function strict_loss(x)
    li = lik(x)
    o = map(i->(rec(x, i) + KL(μ[i](A[i](x)), logσ[i](A[i](x))) - log(α[i])), 1:comp)
    dot(li, o)
end


A = [Dense(nx, nh, swish) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logσ = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, swish), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logσ, f);
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. * logσ)
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))

α = [1 / comp for i in 1:comp] # no prior information
s = 0.06
b = 1 / s

k = 1
@time for i in k:k + 9
    # count the l_ik matrix
    global LI = lik_n(dataT)
    #change_label(LI,X)
    # count α
    global α = sum(LI, dims = 1) / ndat
    #li = [round.(LI[i,:]) for i in 1:ndat]
    #Flux.train!(loss_l, ps, zip(dataT,li), opt)
    Flux.train!(loss, ps, zip(dataT,), opt)
    prumer = round(sum(loss.(dataT)) / ndat, digits = 4)
    max = round(maximum(loss.(dataT)), digits = 4)
    n_vec = sum(round.(LI),dims=1)
    if isnan(prumer)
        println("NaN")
        break
    end
    println("Epoch $i: \nn=$(round.(n_vec)) \navg=$prumer, max=$max")
end

LI1 = LI[:,1]
scatter(X[1,:],X[2,:],zcolor=LI1,aspect_ratio=:equal,label="data classes")

function generate_new(f,n)
    new = []
    for i in 1:comp
        z = rand(MvNormal(zeros(nz),I),n)
        temp = hcat([f(z[:,i]) for i in 1:n]...)


scatter(dt[:,1],dt[:,2],legend=:bottomright,label="data");
scatter!(ts1[:,1],ts1[:,2],label="f1");
scatter!(ts2[:,1],ts2[:,2],label="f2");
scatter!(ts3[:,1],ts3[:,2],label="f3")

gen = vcat(ts1,ts2,ts3)
scatter(dt[:,1],dt[:,2],zcolor=LI[:,1],legend=:bottomright,label="data")
scatter!(gen[:,1],gen[:,2])