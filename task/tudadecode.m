function [Gamma,vpath,error,Ypred] = tudadecode(X,Y,T,tuda,new_experiment,parallel_trials)
% Having estimated the TUDA model (i.e. the corresponding decoding models)
% in the same or a different data set, this function finds the model time
% courses (with no re-estimation of the decoding parameters) 
%
% INPUT
% X: Brain data, (time by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
%               For binary classification problems, Y is (time by 1) and
%               has values -1 or 1
%               For multiclass classification problems, Y is (time by classes) 
%               with indicators values taking 0 or 1. 
%           If the stimulus is the same for all trials, Y can have as many
%           rows as trials, e.g. (trials by q) 
% T: Length of series or trials
% tuda: Estimated TUDA model, using tudatrain
% new_experiment: Whether or not the estimated model is going to be applied
%   on data that follows the same paradigm used to train the model. If the
%   paradigm changes, then new_experiment should be 1. For example, that
%   would be the case if we train the model on perception and test it on
%   recalling. 
% parallel_trials: if set to 1, then 
%   all trials have the same experimental design and that the
%   time points correspond between trials; in this case, all trials
%   must have the same length. If set to 0, then there is not a fixed
%   experimental design for all trials. 
%
% OUTPUT 
% Gamma: Time courses of the states (decoding models) probabilities given data
% vpath: Most likely state path of hard assignments
% error: Error for each state on the new data;
%        if parallel_trials = 1, then error is 
%          (trial tim by no. of stimuli by no. of states);
%        otherwise, this is 
%          (total time by no. of stimuli by no. of states).
% Ypred: Predicted stimulus at each time point
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

if nargin < 5, new_experiment = 0; end
if nargin < 6, parallel_trials = 0; end

max_num_classes = 5;
do_preproc = 1; 

N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2); K = tuda.train.K;

% Check options and put data in the right format
tuda.train.parallel_trials = 0; 
if isfield(tuda.train,'orders')
    orders = tuda.train.orders;
    tuda.train = rmfield(tuda.train,'orders');
else
    orders = [];
end
if isfield(tuda.train,'active')
    active = tuda.train.active;
    tuda.train = rmfield(tuda.train,'active');
else
    active = [];
end

classification = length(unique(Y(:))) < max_num_classes;
if classification
    Ycopy = Y;
    if size(Ycopy,1) == N 
        Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
    end
    % no demeaning by default if this is a classification problem
    if ~isfield(tuda.train,'demeanstim'), tuda.train.demeanstim = 0; end
end

if do_preproc
    if isfield(tuda.train,'embeddedlags'), el = tuda.train.embeddedlags; end
    tuda.train.intercept = 0;
    [X,Y,T,options] = preproc4hmm(X,Y,T,tuda.train); % this demeans Y
    p = size(X,2);
    if classification && length(el) > 1
        Ycopy = reshape(Ycopy,[ttrial N q]);
        Ycopy = Ycopy(-el(1)+1:end-el(end),:,:);
        Ycopy = reshape(Ycopy,[T(1)*N q]);
    end
end
if ~isempty(active), tuda.train.active = active; end 
if ~isempty(orders),  tuda.train.orders = orders;  end 

if isfield(options,'A') % it is done in preproc4hmm
    options = rmfield(options,'A');
end
if isfield(options,'parallel_trials')
    options = rmfield(options,'parallel_trials'); 
end
if isfield(options,'add_noise')
    options = rmfield(options,'add_noise');
end

% Put X and Y together
Ttmp = T;
T = T + 1;
Z = zeros(sum(T),q+p,'single');
for j = 1:N
    t1 = (1:T(j)) + sum(T(1:j-1));
    t2 = (1:Ttmp(j)) + sum(Ttmp(1:j-1));
    if strcmp(options.classifier,'LDA') || (isfield(options,'encodemodel') && options.encodemodel)
        Z(t1(2:end),1:p) = X(t2,:);
        Z(t1(1:end-1),(p+1):end) = Y(t2,:);
    else
        Z(t1(1:end-1),1:p) = X(t2,:);
        Z(t1(2:end),(p+1):end) = Y(t2,:);        
    end
end 

if new_experiment
    off_diagonal = [tuda.P(triu(true(K),1)); tuda.P(tril(true(K),-1))];
    in_diagonal = tuda.P(eye(K)==1);
    tuda.P(triu(true(K),1)) = mean(off_diagonal);
    tuda.P(tril(true(K),-1)) = mean(off_diagonal);
    tuda.P(eye(K)==1) = mean(in_diagonal);    
    tuda.Pi(:) = mean(tuda.Pi);
end

% Run TUDA inference
options.S = -ones(p+q);
if strcmp(options.classifier,'LDA') || (isfield(options,'encodemodel') && options.encodemodel)
    options.S(p+1:end,1:p) = 1;
else
    options.S(1:p,p+1:end) = 1;
end
options.updateObs = 0;
options.updateGamma = 1;
options.updateP = 1; 
options.hmm = tuda; 
options.repetitions = 0;
options.pca = 0; 
options.cyc = 1; 
options.tuda = 0;
[~,Gamma,~,vpath] = hmmmar(Z,T,options);
T = T - 1; 

if parallel_trials
    if classification, error = zeros(max(T),K);
    else, error = zeros(max(T),q,K);
    end
else
    if classification, error = zeros(size(Gamma,1),K);
    else, error = zeros(size(Gamma,1),q,K);
    end    
end
Betas = tudabeta(tuda);
if nargout>2
    for k = 1:K 
      Ypred = X * Betas(:,:,k);
      if classification
          Ypred = continuous_prediction_2class(Ycopy,Ypred);
          Y = continuous_prediction_2class(Ycopy,Y);
          if q == 1
              e = abs(Y - Ypred) < 1e-4;
          else
              e = sum(abs(Y - Ypred),2) < 1e-4;
          end
      else
          e = (Y - Ypred).^2;
      end
      if parallel_trials
          maxT = max(T);
          me = zeros(maxT,1);
          ntrials = zeros(maxT,1);
          for j = 1:N
              t0 = sum(T(1:j-1));
              ind_1 = (1:T(j)) + t0;
              ind_2 = 1:length(ind_1);
              me(ind_2,:) = me(ind_2,:) + e(ind_1,:);
              ntrials(ind_2) = ntrials(ind_2) + 1;
          end
          e = me ./ ntrials;
      end
      if classification, error(:,k) = e;
      else, error(:,:,k) = e;
      end
    end

    error = squeeze(error); 
end

end
