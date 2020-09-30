# functions

linear(x) = x

function loss(x)
   HID = A(x);
   zsample = z(μ(HID),logs(HID));
   0.5*sum((x.-f(zsample)).^2) + s*KL(μ(HID),logs(HID))
end

function run_for(ep)
   @time for i in k:k+ep-1
       Flux.train!(loss,ps,data_train,opt)
       l = loss(dataT[1])
       if isnan(l)
           break
       end
       println("Epoch: $i, loss:$l")  
   end
   global k = k + ep;
   global K = vcat(K,k-1);
end

function run_for_auc(ep,model)
   @unpack A,μ,logs,f = model
   FPR, TPR = [], []
   auc_max, k_max = missing, missing 
   auc_progress = [] 
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
       println(i)
       println(auc_progress)
       if i > 1
         if auc1 > maximum(auc_progress)
            println("doing")
            FPR = fpr
            TPR = tpr
            auc_max = auc1
            k_max = i
            println(auc_max)
         end
         if abs(auc_progress[i-1] - auc1) < 0.0001
            println("no significant change of AUC, training terminated...")
            break
         end
       end
       println("Epoch: $i, loss:$l, auc=$auc1, auc_max=$auc_max")
       auc_progress = vcat(auc_progress,auc1)
   end
   return auc_progress,FPR,TPR,auc_max,k_max
end
   
# I don't know how to use this function inside a loop 
function better_save(auc1,fpr,tpr,i,auc_progress)
   if auc1 > maximum(auc_progress)
      println("doing")
      global FPR = fpr
      TPR = tpr
      auc_max = auc1
      k_max = i
      println(auc_max)
   end
   return FPR,TPR,auc_max,k_max
end


# u jiných než čistě numerických typů to neumí udělat savename
function save_results()
   result = @dict(auc_max,k_max)
   roc_result = @dict(FPR,TPR,auc_progress)
   model = @dict(A, μ, logσ, f)
   safesave(datadir("results", savename(result,"bson")),result)
   safesave(datadir("results", savename("roc",roc_result,"bson")),roc_result)
   safesave(datadir("results", savename("model",model,"bson")),model)
end
