function flatten_hmm(hmm,filename)

K = hmm.K;
train = hmm.train;
prior = hmm.prior;
Dir_alpha = hmm.Dir_alpha;
Pi = hmm.Pi;
Dir2d_alpha = hmm.Dir2d_alpha;
P = hmm.P;

for k = 1:K
    if isfield(hmm.state(k),'W') && isfield(hmm.state(k).W,'Mu_W') && ~isempty(hmm.state(k).W.Mu_W)
        eval(['state_' num2str(k-1) '_Mu_W = hmm.state(k).W.Mu_W;']);
        eval(['state_' num2str(k-1) '_S_W = hmm.state(k).W.S_W;']);
        if isfield(hmm.state(k).W,'iS_W') 
            eval(['state_' num2str(k-1) '_iS_W = hmm.state(k).W.iS_W;']);
        end
    end
    if isfield(hmm.state(k),'Omega') && isfield(hmm.state(k).Omega,'Gam_rate')
        eval(['state_' num2str(k-1) '_Omega_Gam_rate = hmm.state(k).Omega.Gam_rate;']);    
        eval(['state_' num2str(k-1) '_Omega_Gam_shape = hmm.state(k).Omega.Gam_shape;']);
        if isfield(hmm.state(k).Omega,'Gam_irate') 
            eval(['state_' num2str(k-1) '_Omega_Gam_irate = hmm.state(k).Omega.Gam_irate;']); 
        end
        eval(['state_' num2str(k-1) '_prior_Omega_Gam_rate = hmm.state(k).prior.Omega.Gam_rate;']);    
        eval(['state_' num2str(k-1) '_prior_Omega_Gam_shape = hmm.state(k).prior.Omega.Gam_shape;']);
    end
end

if isfield(hmm,'Omega') && isfield(hmm.Omega,'Gam_rate')
    Omega_Gam_rate = hmm.Omega.Gam_rate;    
    Omega_Gam_shape = hmm.Omega.Gam_shape;
    if isfield(hmm.Omega,'Gam_irate')
        Omega_Gam_irate = hmm.Omega.Gam_irate;  
    end
    prior_Omega_Gam_rate = hmm.prior.Omega.Gam_rate;
    prior_Omega_Gam_shape = hmm.prior.Omega.Gam_shape;
end

save_string = strcat("save('",filename,"','K','train','Dir_alpha','Pi','Dir2d_alpha','P'");

for k = 1:K   
    if isfield(hmm.state(k),'W') && isfield(hmm.state(k).W,'Mu_W') && ~isempty(hmm.state(k).W.Mu_W)
        save_string = strcat(save_string,",'state_",num2str(k-1),"_Mu_W'");        
        save_string = strcat(save_string,",'state_",num2str(k-1),"_S_W'");
        if isfield(hmm.state(k).W,'iS_W') 
            save_string = strcat(save_string,",'state_",num2str(k-1),"_iS_W'");
        end
    end
    if isfield(hmm.state(k),'Omega') && isfield(hmm.state(k).Omega,'Gam_rate')
        save_string = strcat(save_string,",'state_",num2str(k-1),"_Omega_Gam_rate'");        
        save_string = strcat(save_string,",'state_",num2str(k-1),"_Omega_Gam_shape'");    
        if isfield(hmm.state(k).Omega,'Gam_irate') 
            save_string = strcat(save_string,",'state_",num2str(k-1),"_Omega_Gam_irate'");
        end
        save_string = strcat(save_string,",'state_",num2str(k-1),"_prior_Omega_Gam_rate'");        
        save_string = strcat(save_string,",'state_",num2str(k-1),"_prior_Omega_Gam_shape'");            
    end
end

if isfield(hmm,'Omega') && isfield(hmm.Omega,'Gam_rate')
    save_string = strcat(save_string,",'Omega_Gam_rate'");
    save_string = strcat(save_string,",'Omega_Gam_shape'");
    if isfield(hmm.Omega,'Gam_irate')    
        save_string = strcat(save_string,",'Omega_Gam_irate'");
    end
    save_string = strcat(save_string,",'prior_Omega_Gam_rate'");
    save_string = strcat(save_string,",'prior_Omega_Gam_shape'");    
end
save_string = strcat(save_string,")");

eval(save_string)

end