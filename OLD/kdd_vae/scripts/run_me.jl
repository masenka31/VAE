function run_exp(par,dataT,ep)
    # unpack parameters of model
    @unpack nx,nz,nh,s,η,opts,activ,run_id = par

    # create the neural networks 
    A, μ, logs = Dense(nx, nh, activ), Dense(nh, nz), Dense(nh, nz);
    f = Chain(Dense(nz,nh,activ),Dense(nh,nx));
    z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ));
    KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. *logs);
    ps = Flux.params(A, μ, logs, f);
    opt = opts(η)

    # loss function (because if it is not defined here, it doesn't work) 
    function loss(x)
        HID = A(x);
        zsample = z(μ(HID),logs(HID));
        0.5*sum((x.-f(zsample)).^2) + s*KL(μ(HID),logs(HID))
    end

    global FPR = [] 
    global TPR = []
    global auc_max, k_max = missing, missing 
    global auc_progress = []

    # training procedure (can't be in a function either) 
    l_prev = loss(dataT[1])
    if isnan(l_prev)
        println("loss NaN before training, process will be terminated")
    else
        global FPR = [] 
        global TPR = []
        global auc_max, k_max = missing, missing 
        global auc_progress = [] 
        l_prev = loss(dataT[1])
        @time for i in 1:ep
            Flux.train!(loss,ps,data_train,opt)
            L = loss.(dataT)
            global l = loss(dataT[1])
            if isnan(l)
              "loss diverged into NaN, training terminated..."
              break
            end
            fpr,tpr = roccurve(L,labels)
            auc1 = auc(fpr,tpr)
            if i > 1
              # this way we save the best model there is throughtout training process 
              if auc1 > maximum(auc_progress)
                 FPR = fpr
                 TPR = tpr
                 auc_max = auc1
                 k_max = i
                 global saved_model = @dict(A,μ,logs,f) 
              end
              if abs(auc_progress[i-1] - auc1) < 0.000001
                 println("no significant change of AUC, training terminated...")
                 break
              end
            end
            println("Epoch: $i, loss:$l, auc=$auc1, auc_max=$auc_max")
            auc_progress = vcat(auc_progress,auc1)
        end
    end

    println("Results: \n maximum AUC = $auc_max")

    @unpack A,μ,logs,f = saved_model

    # ...and save - also cannot be a function
    result = @dict(auc_max,k_max,FPR,TPR,auc_progress,A,μ,logs,f,nx,nz,nh,s,η,opts,activ)
    safesave(datadir("results", savename("(run-$run_id)",result,"bson")),result)

    printstyled("Current experiment completed.\n",bold=true,color=:cyan) 
end

