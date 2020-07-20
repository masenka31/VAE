# load data
include(scriptsdir("init_pulsars.jl"))

# parameters
nx = 8
nh = 20
nh1 = 20
nh2 = 10
nz = 4
s = 1
ep = 1000
η = 0.0001 

# create the neural networks
# A, μ, logs = Dense(nx, nh1, sigmoid), Chain(Dense(nh1,nh2,sigmoid),Dense(nh2,nz)), Chain(Dense(nh1,nh2,sigmoid),Dense(nh2,nz))
A, μ, logs = Dense(nx, nh, sigmoid), Dense(nh,nz), Dense(nh,nz)
f = Chain(Dense(nz, nh, swish), Dense(nh, nx));
# f = Chain(Dense(nz,nh2,sigmoid),Dense(nh2,nh1,sigmoid),Dense(nh1,nx))
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ));
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. * logs);
ps = Flux.params(A, μ, logs, f);
opt = ADAM(η)

# loss function
function loss(x)
    HID = A(x);
    zsample = z(μ(HID), logs(HID));
    0.5 * sum((x .- f(zsample)).^2) + s * KL(μ(HID), logs(HID))
end

rec_K = 50
function rec_loss(x)
    HID = A(x)
    sample = hcat([z(μ(HID), logs(HID)) for i in 1:rec_K]...)
    X = hcat([f(sample[:,i]) for i in 1:rec_K]...)
    Y = vcat([sum((x .- X[:,i]).^2) for i in 1:rec_K]...)
    mean(Y)
end

x = dataN_train[1]

# training procedure
@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global l = loss(x)
    if isnan(l)
        println("loss diverged into NaN, training terminated...")
        break
    end
    L_train = loss.(dataN)
    LA = loss.(dataA)
    avg_norm = mean(L_train)
    avg_an = mean(LA)
    if i%5 == 0
        L = rec_loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr,tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s,η,nh)
        safesave(datadir("ALL_DATA_BEST", savename("run-hard-2", final_model, "bson")), final_model)
        println("AUC: $auc1")
    end
    println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an))")
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 

