function [hmm,Gamma,Xi,fehist,actstates] = hmmtrain(data,T,hmm,Gamma,residuals,fehist)
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
% knocked       states knocked out by the Bayesian inference
%
% hmm.Pi          - intial state probability
% hmm.P           - state transition matrix
% hmm.state(k).$$ - whatever parameters there are in the observation model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T); K = hmm.train.K;
ndim = size(data.X,2); 

if nargin<6, fehist=[]; 
elseif ~isempty(fehist), fprintf('Restarting at cycle %d \n',length(fehist)+1); 
end

actstates = ones(1,K);
cyc_to_go = 0;

for cycle=1:hmm.train.cyc    
    
    if hmm.train.updateGamma,
              
        %%%% E step
        if hmm.K>1 || cycle==1
         
            [Gamma,Gammasum,Xi] = hsinference(data,T,hmm,residuals);
            if size(Gammasum,1)>1, Gammasum = sum(Gammasum); end
            
            if (hmm.K>1 && any(round(Gammasum) >= sum(T)-N*hmm.train.maxorder))
                fprintf('cycle %i: All the points collapsed in one state \n',cycle)
                break
            end
        end
        % any state to remove?
        as1 = find(actstates==1);
        [as,hmm,Gamma,Xi] = getactivestates(data.X,hmm,Gamma,Xi);
        if any(as==0), 
            cyc_to_go = hmm.train.cycstogoafterevent; 
            data.C = data.C(:,as==1);
        end
        actstates(as1(as==0)) = 0;
        
        %%%% Free energy computation
        fehist = [fehist; sum(evalfreeenergy(data.X,T,Gamma,Xi,hmm,residuals))];
        strwin = ''; if hmm.train.meancycstop>1, strwin = 'windowed'; end
        if cycle>(hmm.train.meancycstop+1) && cyc_to_go==0
            chgFrEn = mean( fehist(end:-1:(end-hmm.train.meancycstop+1)) - ...
                fehist(end-1:-1:(end-hmm.train.meancycstop)) )  ...
                / (fehist(1) - fehist(end));
            if hmm.train.verbose, 
                fprintf('cycle %i free energy = %g, %s relative change = %g \n',...
                    cycle,fehist(end),strwin,chgFrEn); 
            end
            if (abs(chgFrEn) < hmm.train.tol), break; end
        elseif hmm.train.verbose, fprintf('cycle %i free energy = %g \n',cycle,fehist(end)); %&& cycle>1
        end
        if cyc_to_go>0, cyc_to_go = cyc_to_go - 1; end
        
    else
        Xi=[]; fehist=0;
    end
    
    %%%% M STEP
       
    % Observation model
    hmm=obsupdate(data.X,T,Gamma,hmm,residuals);
    
    if hmm.train.updateGamma,
        % transition matrices and initial state
        hmm=hsupdate(Xi,Gamma,T,hmm);
    else
        break % one is more than enough
    end

end

if hmm.train.verbose
    fprintf('Model: %d kernels, %d dimension(s), %d data samples \n',K,ndim,sum(T));
end

return;

