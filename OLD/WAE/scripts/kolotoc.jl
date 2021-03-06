nx = 2
nz = 2
nh = 5;
A, μ, logσ = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz, nh, swish), Dense(nh, nx))
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. * logσ)
γ = 0.01
c = 1
# kernel = GaussianKernel(γ) #IMQKernel()
kernel = IMQKernel(c)
n_sample = 200
opt = Flux.ADAM(1e-2)
ps = Flux.params(A, μ, logσ, f)
s = 0.2


@time for i in k:k + 49
    Flux.train!(loss, ps, data_train, opt)
    if i % 10 == 0
        println("Epoch: $i, loss: $(loss(dataT[1]))")
    end
    if isnan(loss(dataT[1]))
        println("Loss in NaN!")
        break
    end
end
test = generate_new(500, nz, f)
rec = reconstruct_data(dataT)
dst = mmd(kernel, X, test)
    
scatter(x, y, aspect_ratio = :equal, label = "X");
plt = scatter!(test[1,:], test[2,:], label = "f(e)")
scatter(x, y, aspect_ratio = :equal, label = "X");
plt2 = scatter!(rec[1,:], rec[2,:], label = "VAE(X)")


e = 250
sv = @dict(e,dst,s,c,A,μ,logσ,f,opt,test,rec)
safesave(datadir("podkova_hand",savename(sv,"bson")),sv)
safesave(plotsdir("podkova_hand",savename("gen",sv,"pdf")),plt)
safesave(plotsdir("podkova_hand",savename("rec",sv,"pdf")),plt2)

