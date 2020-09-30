function run_me(par,dataT,ep)
    @unpack s,nh,nz,γ,n_sample,nx = par
    data_train = zip(dataT,)

    A, μ, logσ = Dense(nx, nh), Dense(nh, nz), Dense(nh, nz)
    f = Chain(Dense(nz,nh),Dense(nh,nx))
    z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
    KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)
    kernel = GaussianKernel(γ) #IMQKernel()
    ps = Flux.params(A, μ, logσ, f)
    opt = ADAM(0.01)
    
    function loss(x)
        HID = A(x);
        zsample = z(μ(HID),logσ(HID))
        sample_rnd = hcat([randn(nz) for i in 1:n_sample]...)
        sample_z = hcat([z(μ(HID),logσ(HID)) for i in 1:n_sample]...)
        0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample_rnd,sample_z)
    end

    @time for i in 1:ep
        Flux.train!(loss,ps,data_train,opt)
        println("Epoch $i")
        if isnan(loss(dataT[1]))
            println("Loss in NaN!")
            break
        end
    end
    
    function generate_new(n,nz,f)
        zs = rand(MvNormal(zeros(nz),I),n)
        new = hcat([f(zs[:,i]) for i in 1:n]...)
        return new
    end

    test = generate_new(500,nz,f)
    scatter(x,y,aspect_ratio=:equal,label="data");
    plt = scatter!(test[1,:],test[2,:],label="new");

    d =[mmd(GaussianKernel(i),X,test) for i in [1,0.1,0.01]]
    println("We are at mmd=$d.")

    result = @dict(s,nh,nz,d,γ,n_sample)
    safesave(datadir("results_otocena", savename(result,"bson")),result)
    safesave(plotsdir("MvNormal_otocena",savename(result,"pdf")),plt)

    printstyled("Current experiment completed.\n",bold=true,color=:cyan)
end