function run_exp(par,dt,ep)
    @unpack run_id,s,nh = par

    ndat = size(dt,1)
    dataT = [dt[i,:] for i in 1:ndat] 

    nx = 2
    nz = 2

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
            vc = -sum(x .- temp.^2)
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

    α = [1 / comp for i in 1:comp] # no prior information
    b = 1 / s

    @time for i in 1:ep
        # count the l_ik matrix
        global LI = lik_n(dataT)
        # count α
        global α = sum(LI, dims = 1) / ndat
        Flux.train!(loss, ps, zip(dataT, ), opt)
        prumer = round(sum(loss.(dataT)) / ndat, digits = 4)
        max = round(maximum(loss.(dataT)), digits = 4)
        n_vec = sum(LI,dims=1)
        if isnan(prumer)
            println("Loss is NaN! Training terminated.")
            break
        end
        println("Epoch $i: \nn=$(round.(n_vec)) \navg=$(round(prumer)), max=$(round(max))")
    end

    global LI = lik_n(dataT)
    n_vec = sum(LI,dims=1)

    result = @dict(run_id,s,nh,n_vec)
    safesave(datadir("results", savename(result,"bson")),result)

    plt = scatter(dt[:,1],dt[:,2],zcolor=LI[:,1],legend=:bottomright,label="");
    safesave(plotsdir(savename(result,"pdf")),plt)

    printstyled("Current experiment completed.\n",bold=true,color=:cyan) 

end

    