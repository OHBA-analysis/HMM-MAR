function [ness,Gamma,crithist] = nesstrain(data,T,ness,Gamma,residuals)
%
% Train Hidden Markov Model using using Variational Framework
%
% INPUTS:
%
% data          observations - a struct with X (time series) and C (classes)
% T             Number of time points for each time series
% ness          NESS structure with options specified in ness.train
% Gamma         Initial state courses
% residuals     in case we train on residuals, the value of those.
%
% OUTPUTS
% ness           estimated NESS model
% Gamma         estimated p(state | data)
% maxchhist     historic of Gamma amount of change across iterations
%
% Author: Diego Vidaurre, 
%         CFIN, Aarhus University / OHBA, University of Oxford (2021)

setxx_ness;
crithist = []; 
%e = Inf; 

% for k = 1:3 
%     ness.state(k).P(1,1) = ness.state(k).P(2,2);
%     ness.state(k).P(1,2) = ness.state(k).P(2,1);
% end
    

for cycle = 1:ness.train.cyc
        
    if ness.train.updateGamma
        %%% E step - state inference
        if cycle > 1 && strcmpi(ness.train.stopcriterion,'ChGamma')
            Gamma0 = Gamma;
        end
        [Gamma,~,Xi] = hsinference(data,T,ness,residuals,[],XX,Gamma); 
%         e0 = e;
%         e = sum(evalfreeenergy_ness(T,Gamma,Xi,ness,residuals,XX));
%         fprintf('+ cycle %i free energy = %.10g, relative change = %g \n',cycle,e,e0-e);

    end
    
    %%% M STEP
    
    % Observation model
    ness = obsupdate_ness(T,Gamma,ness,residuals,XX);
    
%     e0 = e; 
%     e = sum(evalfreeenergy_ness(T,Gamma,Xi,ness,residuals,XX));
%     fprintf('++ cycle %i free energy = %.10g, relative change = %g \n',cycle,e,e0-e);
                    
    % Transition matrices and initial state
    ness = hsupdate_ness(Xi,Gamma,T,ness);
    
%     e0 = e;
%     e = sum(evalfreeenergy_ness(T,Gamma,Xi,ness,residuals,XX));
%     fprintf('+++ cycle %i free energy = %.10g, relative change = %g \n',cycle,e,e0-e);
    
    
    % Stopping conditions and reporting
    if strcmpi(ness.train.stopcriterion,'FreeEnergy')
        % computation of free energy is not exact
        crithist(end+1) = sum(evalfreeenergy_ness(T,Gamma,Xi,ness,residuals,XX));
        fprintf('cycle %i Approx free energy = %.10g \n',cycle,crithist(end));
        if cycle > 1
            chgFrEn = (crithist(end) - crithist(end-1)) ...
                / abs(crithist(1) - crithist(end));
            if (abs(chgFrEn) < ness.train.tol), break; end
        end
        
    elseif strcmpi(ness.train.stopcriterion,'ChGamma')
        if cycle > 1
            crithist(end+1) = mean(sum(abs(Gamma0 - Gamma),2)/2 );
            fprintf('cycle %i mean Gamma change = %.3g \n',...
                cycle,crithist(end));
            if (crithist(end) < ness.train.tol), break; end
        else
            crithist(end+1) = 0;
            fprintf('cycle 1  \n')
        end
    else % log likelihood
        crithist(end+1) = -sum(evalfreeenergy_ness(T,Gamma,Xi,ness,residuals,...
            XX,[0 1 1 0 0]));
        fprintf('cycle %i Approx log likelihood = %.10g \n',cycle,crithist(end));
        if cycle > 1
            chL = (crithist(end) - crithist(end-1)) ...
                / abs(crithist(1) - crithist(end));
            if (abs(chL) < ness.train.tol), break; end
         end
    end
    
    % plot state time courses if requested
    if ness.train.plotGamma > 0
        figure(100);clf(100);
        if ness.train.plotGamma == 1 % continuous data
            plot_Gamma (Gamma,T,1);
        elseif ness.train.plotGamma == 2 % full plot
            plot_Gamma (Gamma,T,0);
        end
        drawnow
    end
    
    if ~ness.train.updateGamma
        break % one iteration is enough
    end

end

for k = 1:K
    if isfield(ness.state(k),'cache')
        ness.state(k) = rmfield(ness.state(k),'cache');
    end
end

if ness.train.verbose
    str = 'NESS '; str2 = 'chains';
    if ~isfield(ness.train,'distribution') || strcmp(ness.train.distribution,'Gaussian')
        fprintf('%s Model: %d %s, %d data samples, order %d \n', ...
            str,K,str2,sum(T),ness.train.order);
    elseif strcmp(ness.train.distribution,'logistic')
        fprintf('%s Model: %d %s, %d data samples, logistic regression model. \n', ...
            str,K,str2,sum(T));
    end
    if ness.train.useMEX==0
        fprintf('MEX file was not used \n')
    else
        fprintf('MEX file was used for acceleration \n')
    end
end
    
end

