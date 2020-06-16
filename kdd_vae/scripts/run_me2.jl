function run_exp(par,dataT,ep)
    @unpack nx,nz,nh,s,η,opts,activ = par

    A, μ, logs = Dense(nx, nh, activ), Dense(nh, nz), Dense(nh, nz);
    f = Chain(Dense(nz,nh,activ),Dense(nh,nx));
    z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ));
    KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. *logs);
    #ps = Flux.params(A, μ, logs, f);
    opt = opts(η)

    model = @dict(A,μ,logs,f) 

    l_prev = loss(dataT[1])
    if isnan(l_prev)
        println("loss NaN before training, process will be terminated")
    else
        auc_progress,FPR,TPR,auc_max,k_max = run_for_auc(10,model)
    end

    println("Results: \n maximum AUC = $auc_max")
    save_results()
end