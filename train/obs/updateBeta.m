function hmm = updateBeta(hmm)

K = length(hmm.state); ndim = hmm.train.ndim;
M = hmm.train.pcapred;

for k = 1:K
    if ~hmm.train.active(k), continue; end
    setstateoptions;
    
    if isempty(orders) || train.uniqueAR || ~isempty(train.prior), continue; end
    % shape
    hmm.state(k).beta.Gam_shape = hmm.state(k).prior.beta.Gam_shape + 0.5;
    % rate - mean(W)
    hmm.state(k).beta.Gam_rate = hmm.state(k).prior.beta.Gam_rate + ...
        hmm.state(k).W.Mu_W(1+(~train.zeromean):end,:).^2;
    % rate - cov(W)
    if strcmp(train.covtype,'full') || strcmp(train.covtype,'uniquefull')
        for n1 = 1:M
            for n2 = 1:ndim
                index = (n1-1) * ndim + n2 + (~train.zeromean)*ndim;
                hmm.state(k).beta.Gam_rate(n1,n2) = hmm.state(k).beta.Gam_rate(n1,n2) + ...
                    0.5 * diag(hmm.state(k).W.S_W(index,index) );
            end
        end
    else
        for n1 = 1:M
            for n2 = 1:ndim
                if ndim==1
                    hmm.state(k).beta.Gam_rate(n1,n2) = hmm.state(k).beta.Gam_rate(n1,n2) + ...
                        0.5 * hmm.state(k).W.S_W(n1+(~train.zeromean),n1+(~train.zeromean));
                else
                    hmm.state(k).beta.Gam_rate(n1,n2) = hmm.state(k).beta.Gam_rate(n1,n2) + ...
                        0.5 * hmm.state(k).W.S_W(n2,n1+(~train.zeromean),n1+(~train.zeromean));
                end
            end
        end
    end
end
end