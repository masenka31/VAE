nx = 2
nz = 2
nh = 5;
A, μ, logσ = Dense(nx, nh), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz,nh),Dense(nh,nx))
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)
γ = 0.01
kernel = GaussianKernel(γ) #IMQKernel()  
n_sample = 200
opt = Flux.ADAM(1e-2)
ps = Flux.params(A, μ, logσ, f)
# s = 1

k = 1
for r in 1:4
    @time for i in k:k+49
        Flux.train!(loss,ps,data_train,opt)
        if i%10 == 0
            println("Epoch: $i, loss: $(loss(dataT[1]))")
        end
        if isnan(loss(dataT[1]))
            println("Loss in NaN!")
            break
        end
    end
    global k = k+50
    e = k - 1
    test = generate_new(500,nz,f)
    rec = reconstruct_data(dataT)
    dst = mmd(kernel,X,test)
    sv = @dict(e,dst,s,γ,A,μ,logσ,f,opt,test,rec)
    if r == 4
        scatter(x,y,aspect_ratio=:equal,label="X")
        plt = scatter!(test[1,:],test[2,:],label="VAE(X)")
        safesave(plotsdir("normalni",savename(sv,"pdf")),plt)
    end
    safesave(datadir("res_normalni",savename(sv,"bson")),sv)
end