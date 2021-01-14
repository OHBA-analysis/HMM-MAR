function [hmm,Gamma,Xi,fehist] = hmmtrain(data,T,hmm,Gamma,residuals,fehist)
%
% Train Hidden Markov Model using using Variational Framework
%
% INPUTS:
%
% data          observations - a struct with X (time series) and C (classes)
% T             Number of time points for each time series
% hmm           hmm structure with options specified in hmm.train
% Gamma         Initial state courses
% residuals     in case we train on residuals, the value of those.
%
% OUTPUTS
% hmm           estimated HMMMAR model
% Gamma         estimated p(state | data)
% Xi            joint probability of past and future states conditioned on data
% fehist        historic of the free energies across iterations
%
% hmm.Pi          - intial state probability
% hmm.P           - state transition matrix
% hmm.state(k).$$ - whatever parameters there are in the observation model
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

if nargin<6, fehist=[]; end
cyc_to_go = 0;
setxx;
additiveHMM = hmm.train.additiveHMM;

do_clustering = isfield(hmm.train,'cluster') && hmm.train.cluster;
if do_clustering
    root_P = 0.75;
    hmm.P = root_P * ones(hmm.K);
    hmm.P(eye(hmm.K)==1) = 1;
end

% figure(5);clf(5)
% area(Gamma(1:1000,:));
% drawnow
% pause(0.5)
% % G0= Gamma;

for cycle = 1:hmm.train.cyc
    
    if hmm.train.updateGamma
                
        %%%% E step
        if hmm.K>1 || cycle==1
            % state inference
            if additiveHMM
                [Gamma,~,Xi] = hsinference(data,T,hmm,residuals,[],XX,Gamma);
            else
                [Gamma,~,Xi] = hsinference(data,T,hmm,residuals,[],XX);
            end
            status = checkGamma(Gamma,T,hmm.train);
            
            % check local minima
            epsilon = 1;
            while status == 1
                hmm = hmmperturb(hmm,epsilon);
                disp('Stuck in bad local minima - perturbing the model and retrying...')
                [Gamma,~,Xi] = hsinference(data,T,hmm,residuals,[],XX);
                status = checkGamma(Gamma,T,hmm.train);
                epsilon = epsilon * 2;
            end
            
            % any state to remove?
            [as,hmm,Gamma,Xi] = getactivestates(hmm,Gamma,Xi);
            if hmm.train.dropstates
                if any(as==0)
                    cyc_to_go = hmm.train.cycstogoafterevent;
                    data.C = data.C(:,as==1);
                    [Gamma,~,Xi] = hsinference(data,T,hmm,residuals,[],XX);
                    checkGamma(Gamma,T,hmm.train);
                end
                if sum(hmm.train.active)==1
                    if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic')
                        fehist(end+1) = sum(evalfreeenergylogistic(T,Gamma,Xi,hmm,residuals,XX));
                    elseif additiveHMM
                        fehist(end+1) = sum(evalfreeenergy_addHMM(data.X,T,Gamma,Xi,hmm,residuals,XX));
                    else
                        fehist(end+1) = sum(evalfreeenergy(data.X,T,Gamma,Xi,hmm,residuals,XX));
                    end
                    if hmm.train.verbose
                        fprintf('cycle %i: All the points collapsed in one state, free energy = %g \n',...
                            cycle,fehist(end));
                    end
                    K = 1; break
                end
            end
            
            if hmm.train.plotAverageGamma
                figure(100);clf(100);
                if isfield(hmm.train,'tuda') && hmm.train.tuda
                    plot_Gamma(Gamma,T);
                else
                    if hmm.train.order > 0, dg = hmm.train.order; end
                    if length(hmm.train.embeddedlags) > 1
                        dg = -hmm.train.embeddedlags(1) + hmm.train.embeddedlags(end);
                    end
                    plot(double((1:(T(1)-dg)))/hmm.train.Fs,...
                        squeeze(mean(reshape(Gamma,[(T(1)-dg) length(T) size(Gamma,2)]),2)),...
                        'LineWidth',3)
                    ylim([0 1]); xlim(double([1 (T(1)-dg)])/hmm.train.Fs)
                end
                drawnow
            end
            
        end
       
        %%%% Free energy computation
        if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic')
            fehist(end+1) = sum(evalfreeenergylogistic(T,Gamma,Xi,hmm,residuals,XX));
        elseif additiveHMM
            fehist(end+1) = sum(evalfreeenergy_addHMM(data.X,T,Gamma,Xi,hmm,residuals,XX));
        else
            fehist(end+1) = sum(evalfreeenergy(data.X,T,Gamma,Xi,hmm,residuals,XX));
        end
        strwin = ''; if hmm.train.meancycstop>1, strwin = ' windowed'; end
        if length(fehist) > (hmm.train.meancycstop+1)
            chgFrEn = mean( fehist(end:-1:(end-hmm.train.meancycstop+1)) - ...
                fehist(end-1:-1:(end-hmm.train.meancycstop)) )  ...
                / abs(fehist(1) - fehist(end));
            if hmm.train.verbose
                fprintf('cycle %i free energy = %.10g,%s relative change = %g \n',...
                    cycle,fehist(end),strwin,chgFrEn);
            end
            if (abs(chgFrEn) < hmm.train.tol) && cyc_to_go==0
                break;
            end
        elseif hmm.train.verbose && (length(fehist) == (hmm.train.meancycstop+1))
            chgFrEn = mean( fehist(end:-1:(end-hmm.train.meancycstop+1)) - ...
                fehist(end-1:-1:(end-hmm.train.meancycstop)) ) ;
            fprintf('cycle %i free energy = %.10g,%s absolute change = %g \n',...
                cycle,fehist(end),strwin,chgFrEn);
        elseif hmm.train.verbose
            fprintf('cycle %i free energy = %g \n',cycle,fehist(end)); %&& cycle>1
        end
        if cyc_to_go>0, cyc_to_go = cyc_to_go - 1; end
        
    else
        Xi = []; fehist = 0;
    end
    
    %%%% M STEP
    
    % Observation model
    if hmm.train.updateObs
        setxx
        if additiveHMM
            hmm = obsupdate_addHMM(T,Gamma,hmm,residuals,XX);
        else
            hmm = obsupdate(T,Gamma,hmm,residuals,XX,XXGXX);
        end
    end
    
    % Transition matrices and initial state
    if hmm.train.updateP
        if additiveHMM
            hmm = hsupdate_addHMM(Xi,Gamma,T,hmm);
        else
            hmm = hsupdate(Xi,Gamma,T,hmm);
        end
    end

    if ~hmm.train.updateGamma
        break % one iteration is enough
    end
    
    % some breaking conditions
    if ~additiveHMM
        if  (sum(hmm.train.updateObs + hmm.train.updateGamma + hmm.train.updateP) < 2)
            break % one iteration is enough
        end
        if (hmm.train.maxFOth < Inf)
            if max(getMaxFractionalOccupancy(Gamma,T,hmm.train)) > hmm.train.maxFOth
                disp('Training has been stopped for reaching the threshold of maximum FO')
                break
            end
        end
    end
    
    if hmm.train.tudamonitoring
        hmm.tudamonitor.synch(cycle+1,:) = getSynchronicity(Gamma,T);
        which_x = (hmm.train.S(1,:) == -1);
        which_y = (hmm.train.S(1,:) == 1);
        hmm.tudamonitor.accuracy(cycle+1,:) = ...
            getAccuracy(residuals(:,which_x),residuals(:,which_y),T,Gamma,[],0);
        if ~isempty(hmm.train.behaviour)
            fs = fields(hmm.train.behaviour);
            for ifs = 1:length(fs)
                f = hmm.tudamonitor.behaviour.(fs{ifs});
                y = hmm.train.behaviour.(fs{ifs});
                f(cycle+1,:) = getBehAssociation(Gamma,y,T);
                hmm.tudamonitor.behaviour.(fs{ifs}) = f;
            end
        end
    end
    
    if do_clustering
        if cycle < 20
            hmm.P = root_P^(cycle+1) * ones(hmm.K);
            hmm.P(eye(hmm.K)==1) = 1;
        else
            hmm.P = eye(K);
        end
    end
 
end

for k = 1:K
    if isfield(hmm.state(k),'cache')
        hmm.state(k) = rmfield(hmm.state(k),'cache');
    end
end

if hmm.train.tudamonitoring
    if (abs(chgFrEn) < hmm.train.tol) && cyc_to_go==0
        cycle = cycle - 1;
    end
    hmm.tudamonitor.synch = hmm.tudamonitor.synch(1:cycle+1,:);
    hmm.tudamonitor.accuracy = hmm.tudamonitor.accuracy(1:cycle+1,:);
    if ~isempty(hmm.train.behaviour)
        fs = fields(hmm.train.behaviour);
        for ifs = 1:length(fs)
            f = hmm.tudamonitor.behaviour.(fs{ifs});
            f = f(1:cycle+1,:);
            hmm.tudamonitor.behaviour.(fs{ifs}) = f;
        end
    end
end

if hmm.train.verbose
    if hmm.train.additiveHMM, str = 'Additive HMM '; str2 = 'chains';
    else, str = 'HMM '; str2 = 'states';
    end
    if ~isfield(hmm.train,'distribution') || strcmp(hmm.train.distribution,'Gaussian')
        fprintf('%s Model: %d %s, %d data samples, covariance: %s, order %d \n', ...
            str,K,str2,sum(T),hmm.train.covtype,hmm.train.order);
    elseif strcmp(hmm.train.distribution,'logistic')
        fprintf('%s Model: %d %s, %d data samples, logistic regression model. \n', ...
            str,K,str2,sum(T));
    end
    if hmm.train.useMEX==0
        fprintf('MEX file was not used \n')
    else
        fprintf('MEX file was used for acceleration \n')
    end
end

if do_clustering
    kk = [];
    Gamma0 = Gamma;
    for k = 1:hmm.K
        t = find(sum(Gamma0,2)>0,1);
        [~,m] = max(Gamma0(t,:));
        if isempty(m)
            break;
        end
        kk = [kk m];
        Gamma0(:,m) = 0;
    end
    if length(kk) < hmm.K
        kk = [kk setdiff((1:hmm.K), kk)];
    end
    Gamma = Gamma(:,kk);
    hmm.state = hmm.state(kk);
    hmm.Pi = hmm.Pi(kk);
    hmm.Dir_alpha = hmm.Dir_alpha(kk);
end

end

