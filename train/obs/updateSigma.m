function hmm = updateSigma(hmm)

K = length(hmm.state); ndim = hmm.train.ndim;
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else Q = ndim; end
if Q==1, return; end

setstateoptions;
for k=1:K
    if ~hmm.train.active(k), continue; end
    if isempty(orders) || train.uniqueAR || ~isempty(train.prior), continue; end
    %shape
    if train.symmetricprior
        hmm.state(k).sigma.Gam_shape = hmm.state(k).prior.sigma.Gam_shape + length(orders);
        for n=1:ndim
            hmm.state(k).sigma.Gam_shape(n,n) = hmm.state(k).prior.sigma.Gam_shape(n,n) + 0.5*length(orders);
        end
    else
        hmm.state(k).sigma.Gam_shape = hmm.state(k).prior.sigma.Gam_shape + 0.5*length(orders);
    end
    %rate
    hmm.state(k).sigma.Gam_rate = hmm.state(k).prior.sigma.Gam_rate;
    % mean(W)
    for n1=1:Q
        if any(S(n1,:)==1)
            for n2=find(S(n1,:)==1)
                if train.symmetricprior && n1>n2
                    continue;
                end
                index = n1 + (0:length(orders)-1)*Q + ~train.zeromean;
                hmm.state(k).sigma.Gam_rate(n1,n2) = hmm.state(k).sigma.Gam_rate(n1,n2) + ...
                    0.5 * (hmm.state(k).W.Mu_W(index,n2)' * ...
                    ((hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate') .* hmm.state(k).W.Mu_W(index,n2)) );
                if hmm.train.symmetricprior && n1~=n2
                    index = n2 + (0:length(orders)-1)*ndim + ~train.zeromean;
                    hmm.state(k).sigma.Gam_rate(n1,n2) = hmm.state(k).sigma.Gam_rate(n1,n2) + ...
                        0.5 * (hmm.state(k).W.Mu_W(index,n1)' * ...
                        ((hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate') .* hmm.state(k).W.Mu_W(index,n1)));
                end
            end
        end
    end
    % cov(W)
    if strcmp(train.covtype,'full') || strcmp(train.covtype,'uniquefull')
        for n1=1:Q
            if any(S(n1,:)==1)
                for n2=find(S(n1,:)==1)
                    if train.symmetricprior && n1>n2
                        continue;
                    end
                    if all(S(:)==1)
                        index = (0:length(orders)-1) * (ndim*Q) + (n1-1) * ndim + n2 + (~train.zeromean)*ndim;
                    else
                        index = false(size(S));
                        index(n1,n2) = true;
                        index = index(:);
                    end
                    hmm.state(k).sigma.Gam_rate(n1,n2) = hmm.state(k).sigma.Gam_rate(n1,n2) + ...
                        0.5 * sum((hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate') .* ...
                        diag(hmm.state(k).W.S_W(index,index) ));
                    if hmm.train.symmetricprior && n1~=n2
                        index = (0:length(orders)-1) * ndim^2 + (n2-1) * ndim + n1 + (~train.zeromean)*ndim;
                        hmm.state(k).sigma.Gam_rate(n1,n2) = hmm.state(k).sigma.Gam_rate(n1,n2) + ...
                            0.5 * sum((hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate') .* ...
                            diag(hmm.state(k).W.S_W(index,index) ));
                    end
                end
            end
        end

    else
        for n1=1:Q
            if any(S(n1,:)==1)
                for n2=find(S(n1,:)==1)
                    if train.symmetricprior && n1>n2
                        continue;
                    end
                    index = n1 + (0:length(orders)-1)*Q + ~train.zeromean;
                    hmm.state(k).sigma.Gam_rate(n1,n2) = hmm.state(k).sigma.Gam_rate(n1,n2) + ...
                        0.5 * sum((hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate') .* ...
                        diag( permute(hmm.state(k).W.S_W(n2,index,index),[2 3 1]) )) ;
                    if train.symmetricprior && n1~=n2
                        index = n2 + (0:length(orders)-1)*ndim + ~train.zeromean;
                        hmm.state(k).sigma.Gam_rate(n1,n2) = hmm.state(k).sigma.Gam_rate(n1,n2) + ...
                            0.5 * sum((hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate') .* ...
                            diag( permute(hmm.state(k).W.S_W(n1,index,index),[2 3 1]) )) ;
                        hmm.state(k).sigma.Gam_rate(n2,n1) = hmm.state(k).sigma.Gam_rate(n1,n2);
                    end
                end
            end
        end
    end
end
end