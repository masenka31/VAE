# include
include(scriptsdir("anomaly_detection", "emvae_functions.jl"))
include(scriptsdir("anomaly_detection", "init_kdd.jl"))

# initialize network
nx = 22
nz = 10
nh = 30
comp = 5
α = [1 / comp for i in 1:comp] # no prior information
s = 300
b = 1 / s
ep = 200

opt = ADAM(0.0005)
A = [Dense(nx, nh, σ) for i in 1:comp]
μ = [Dense(nh, nz) for i in 1:comp]
logs = [Dense(nh, nz) for i in 1:comp]
f = [Chain(Dense(nz, nh, σ), Dense(nh, nx)) for i in 1:comp]

ps = Flux.params(A, μ, logs, f);
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs)
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))

function loss(x)
    li = lik(x)
    o = map(i -> (rec(x, i) + KL(μ[i](A[i](x)), logs[i](A[i](x))) - log(α[i] + 10^(-10))), 1:comp)
    dot(li, o)
end

# training procedure
x = dataN_train[1]

global LI = lik_n(dataN)                            # count the l_ik matrix
global α = sum(LI, dims=1) / normal_count         # count α

@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global LI = lik_n(dataN)                        # count the l_ik matrix
    global α = sum(LI, dims=1) / normal_count     # count α
    global l = loss(x)
    if isnan(l)
        println("loss diverged into NaN, training terminated...")
        break
    end
    # counts mean loss for normal and anomaly data
    L_train = loss.(dataN)
    LA = loss.(dataA)
    avg_norm = mean(L_train)
    avg_an = mean(LA)
    # every 5 epochs saves current model
    if i % 5 == 0
        L = loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr, tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s,α)
        safesave(datadir("EMVAE_KDD", savename("run-1", final_model, "bson")), final_model)
        println("AUC: $auc1")
    end
    n_vec = sum(round.(LI), dims=1)
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an)), n: $n_vec")
end

printstyled("Current experiment completed.\n", bold=true, color=:cyan)
