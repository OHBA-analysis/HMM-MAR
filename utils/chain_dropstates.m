function [pruned_hmm,fe,fehist] = chain_dropstates(hmm,X,T,verbose)
% pruned_hmm: best model of the chain
% fe: free energy of pruned_hmm
% fehist: history of all the chain

K = length(hmm.state);
fehist = []; fe = []; pruned_hmm = hmm;

if nargin<4, verbose = 0; end

if K==2
    return
end

fe = Inf;
for k = 1:K
    [tmp_hmm,tmp_fe] = dropstate(hmm,k,X,T);
    if ~isempty(tmp_fe) && tmp_fe<fe
        fe = tmp_fe; pruned_hmm = tmp_hmm;
        if verbose
            fprintf('hmm is taken with %d states \n',K-1)
        end
    end
end

[tmp_hmm,tmp_fe,fehist] = chain_dropstates(pruned_hmm,X,T,verbose);

fehist = [fe fehist];
if ~isempty(tmp_fe) && tmp_fe<fe
    fe = tmp_fe; pruned_hmm = tmp_hmm;
end
    
end


