function tests = specttest(sp_fit,Nperm,within_state,relative)
%
% From a multitaper or MAR estimation of the states spectra (per subject),  
% or from a spectral factorisation as returned by spectdecompose,
% this function performs, for each state and frequency component (or bin), 
% statistical testing (via permutations) of power/coherence.
% Testing can be performed in two different ways:
% 1) Is power/coh higher or lower for that state than for the other states?
% 2) Is each value of power/coh higher or lower than the other values 
%       in this state?
% Multiple corrections is applied; it is done separately for coherence
% and power, and separately for both sides of the test (lower or higher),
% separately for each state (Benjamini-Hochberg correction), 
% and separately for each frequency bin / frequency component. 
% If relative is true, the testing is done over the relative pow/coh of
% each connection with regard to the average level of the state.
% This could be useful for example if there is one state with generalised
% higher power/coherence, but we want to see if the patterns of the other
% states are significative. If we test absolute values (relative==0)
% only the dominant state will exhibit significance. 
%
% INPUTS: 
%
% sp_fit                The output of hmmspectramar or hmmspectramt, or the
%                       output of spectdecompose, i.e. a cell, 
%                       where each element corresponds to one subject
% Nperm                 No. of permutations
% within_state          If 1, test each value of PSD/Coh  
%                           against the rest of the values within each state 
%                       If 0, test each value of PSD/Coh across states
% relative              Test for relative changes within state? (default 0) 
%
% OUTPUT:
% 
% tests                 Struct with 4 fields: lower, higher, lower_corr, higher_corr
%                       ('_corr' reflects to multiple comparisons correction).
%                       Each field has a field 'states', just like each
%                       element of sp_fit, with fields state(k).psd and 
%                       state(k).coh containing p-values for PSD and coherence
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin < 3, within_state = 1; end
if nargin < 4, relative = 0; end

tests = struct();
tests.higher = struct();
tests.higher.state = struct();
tests.higher_corr = struct();
tests.higher_corr.state = struct();
tests.lower = struct();
tests.lower.state = struct();
tests.lower_corr = struct();
tests.lower_corr.state = struct();

N = length(sp_fit);
K = length(sp_fit{1}.state); 
[Nf,ndim,~] = size(sp_fit{1}.state(1).psd);

% PSD
p = ndim;
X = zeros(N,K,p,Nf); 
for n = 1:N
    for nf = 1:Nf
        for k = 1:K
            d = sp_fit{n}.state(k).psd(nf,eye(ndim)==1)';
            X(n,k,:,nf) = d(:);
        end
    end
end
if relative
    for nf = 1:Nf
        m = squeeze(mean(X(:,:,:,nf),2)); % N by p
        for k = 1:K
            X(:,k,:,nf) = permute(X(:,k,:,nf),[1 3 2 4]) - m;
        end
    end
end
if within_state
    pval = permtest1(X,Nperm); % K x p x Nf x 2
else
    pval = permtest2(X,Nperm); % K x p x Nf x 2
end
for k = 1:K % unfold
    tests.lower.state(k).psd = NaN(Nf,ndim,ndim);
    tests.higher.state(k).psd = NaN(Nf,ndim,ndim);
    for j = 1:ndim
        tests.lower.state(k).psd(:,j,j) = pval(k,j,:,1);
        tests.higher.state(k).psd(:,j,j) = pval(k,j,:,2);
    end
end

% PSD corrected
pval = bh(pval);
for k = 1:K % unfold
    tests.lower_corr.state(k).psd = NaN(Nf,ndim,ndim);
    tests.higher_corr.state(k).psd = NaN(Nf,ndim,ndim);
    for j = 1:ndim
        for nf = 1:Nf
            tests.lower_corr.state(k).psd(:,j,j) = pval(k,j,:,1);
            tests.higher_corr.state(k).psd(:,j,j) = pval(k,j,:,2);
        end
    end
end

if ~isfield(sp_fit{1}.state(1),'coh'), return; end

% Coh
ind2 = triu(true(ndim),1);
p = ndim * (ndim-1) / 2; 
X = zeros(N,K,p,Nf); 
for n = 1:N
    for nf = 1:Nf
        for k = 1:K
            d = sp_fit{n}.state(k).coh(nf,ind2)';
            X(n,k,:,nf) = d(:);
        end
    end
end
if relative
    for nf = 1:Nf
        m = squeeze(mean(X(:,:,:,nf),2)); % N by p
        for k = 1:K
            X(:,k,:,nf) = permute(X(:,k,:,nf),[1 3 2 4]) - m;
        end
    end
end
if within_state
    pval = permtest1(X,Nperm); % K x p x Nf x 2
else
    pval = permtest2(X,Nperm); % K x p x Nf x 2
end
for k = 1:K % unfold
    tests.lower.state(k).coh = NaN(Nf,ndim,ndim);
    tests.higher.state(k).coh = NaN(Nf,ndim,ndim);
    for nf = 1:Nf
        d = pval(k,:,nf,1); 
        C = zeros(ndim); C(ind2) = d; C = C + C'; C(eye(ndim)==1) = NaN;
        tests.lower.state(k).coh(nf,:,:) = C;
        d = pval(k,:,nf,2); 
        C = zeros(ndim); C(ind2) = d; C = C + C'; C(eye(ndim)==1) = NaN;
        tests.higher.state(k).coh(nf,:,:) = C;
    end
end

% Coh corrected
pval = bh(pval);
for k = 1:K % unfold
    tests.lower_corr.state(k).coh = NaN(Nf,ndim,ndim);
    tests.higher_corr.state(k).coh = NaN(Nf,ndim,ndim);
    for nf = 1:Nf
        d = pval(k,:,nf,1); 
        C = zeros(ndim); C(ind2) = d; C = C + C'; C(eye(ndim)==1) = NaN;
        tests.lower_corr.state(k).coh(nf,:,:) = C;
        d = pval(k,:,nf,2); 
        C = zeros(ndim); C(ind2) = d; C = C + C'; C(eye(ndim)==1) = NaN;
        tests.higher_corr.state(k).coh(nf,:,:) = C;
    end
end

end


function pval = permtest1(X,Nperm)
% permutation testing routine within states
% Nperm: no. of permutations
[N,K,p,Nf] = size(X);
grotperms = zeros(Nperm,K,Nf,p);
X = permute(X,[1 2 4 3]); % N x K x Nf x p
for perm=1:Nperm
    if perm==1
        Xin = X;
    else
        Xin = zeros(size(X));
        for n = 1:N
            Xin(n,:,:,:) = X(n,:,:,randperm(p,p));
        end
    end
    for j = 1:p
        jj = setdiff(1:p,j);
        grotperms(perm,:,:,j) = sum(Xin(:,:,:,j) - mean(Xin(:,:,:,jj),4),1);
    end
end
pval = zeros(K,p,Nf,2);
for j = 1:p
    for nf = 1:Nf
        for k = 1:K
            pval(k,j,nf,1) = sum(grotperms(:,k,nf,j)<=grotperms(1,k,nf,j)) / (Nperm+1); % lower than
            pval(k,j,nf,2) = sum(grotperms(:,k,nf,j)>=grotperms(1,k,nf,j)) / (Nperm+1); % higher than
        end
    end
end
end


function pval = permtest2(X,Nperm)
% permutation testing routine across states
% Nperm: no. of permutations
[N,K,p,Nf] = size(X);
grotperms = zeros(Nperm,p,Nf,K);
X = permute(X,[1 3 4 2]); % N x p x Nf x K
for perm=1:Nperm
    if perm==1
        Xin = X;
    else
        Xin = zeros(size(X));
        for n = 1:N
            Xin(n,:,:,:) = X(n,:,:,randperm(K,K));
        end
    end
    for k = 1:K
        kk = setdiff(1:K,k);
        grotperms(perm,:,:,k) = sum(Xin(:,:,:,k) - mean(Xin(:,:,:,kk),4),1);
    end
end
pval = zeros(K,p,Nf,2);
for j = 1:p
    for nf = 1:Nf
        for k = 1:K
            pval(k,j,nf,1) = sum(grotperms(:,j,nf,k)<=grotperms(1,j,nf,k)) / (Nperm+1); % lower than
            pval(k,j,nf,2) = sum(grotperms(:,j,nf,k)>=grotperms(1,j,nf,k)) / (Nperm+1); % higher than
        end
    end
end
end


function pval = bh(pval)
% benjamini-hochberg correction across regions
for i = 1:2
    for k = 1:size(pval,1)
        for nf = 1:size(pval,3)
            p = pval(k,:,nf,i)';
            pval(k,:,nf,i) = mafdr(p,'BHFDR',true);
        end
    end
end
end


