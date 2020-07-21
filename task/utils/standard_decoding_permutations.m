function [cv_acc_perm,acc_perm,genplot_perm] = standard_decoding_permutations(X,Y,T,options,binsize,nperms)
% This function computes a null distribution for decoding by randomly
% permuting the labels nperms times. This code is designed to be run
% immediately following a call to standard_decoding, such that the data and
% options are exactly unchanged.
%
% The only additional parameter is nperms, which specifies the number of
% label permutations to be run (default value is 50).
%
% The outputs follow the same format of the outputs of a call to
% standard_decoding, except that they have one additional dimension
% containing nperms different permutation outputs.
% 
% Author: Cam Higgins, OHBA

if nargin < 4 || isempty(options), options = struct(); end
if nargin < 5 || isempty(binsize), binsize = 1; end
if nargin < 6, nperms = 500; end

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
    [cv_acc_perm(:,:,i),acc_perm(:,:,i),genplot_perm(:,:,i)] = standard_decoding(X,Y_perm,T,options,binsize);
    if mod(i,10)==0
        fprintf(['\nPermutation test ',int2str(i),' of ',int2str(nperms),' completed']);
    end
end



end