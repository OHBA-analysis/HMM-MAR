function [acc,acc_star] = tudacv_permutations(X,Y,T,options,nperms)
% This function computes a null distribution for decoding by randomly
% permuting the labels nperms times. This code is designed to be run
% immediately following a call to tudacv, such that the data and
% options are exactly unchanged.
%
% The only additional parameter is nperms, which specifies the number of
% label permutations to be run (default value is 50).
%
% The outputs follow the same format of the outputs of a call to
% tudacv, except that they have one additional dimension
% containing nperms different permutation outputs.
% 
% Author: Cam Higgins, OHBA

if nargin < 4 || isempty(options), options = struct(); end
if nargin < 6 || isempty(nperms), nperms = 50; end

N = length(T); ttrial = T(1); 

if size(Y,1) < size(Y,2); Y = Y'; end
q = size(Y,2);

if size(Y,1) == length(T) % one value per trial
    responses = repelem(Y,[ttrial,1]);
    responses = reshape(responses,[ttrial,N,q]);
elseif length(Y(:)) ~= (ttrial*N*q)
    error('Incorrect dimensions in Y')
else
    responses = reshape(Y,[ttrial,N,q]);
end

for i = 1:nperms
    perm = randperm(N);
    Y_perm = reshape(responses(:,perm,:),[ttrial*N,q]);
    options.verbose = false;
    [acc(:,i),acc_star(:,:,i)] = tudacv(X,Y_perm,T,options);
    if mod(i,10)==0
        fprintf(['\nPermutation test ',int2str(i),' of ',int2str(nperms),' completed']);
    end
end




end