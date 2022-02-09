function ehmm = updateAlpha_ehmm(ehmm)
setstateoptions;
for k = 1:ehmm.K
    for n = 1:ehmm.train.ndim
        if ~regressed(n), continue; end
        indr = S(n,:)==1;
        for i = 1:length(orders)
            index = (i-1)*ehmm.train.ndim + n + ~train.zeromean;
            if ehmm.train.ndim > 1
                sigmaprior = (ehmm.state(k).sigma.Gam_shape(n,indr) ./ ...
                    ehmm.state(k).sigma.Gam_rate(n,indr));
            else
                sigmaprior = zeros(1,sum(indr));
            end
            ehmm.state(k).alpha.Gam_rate(i) = ehmm.state(k).alpha.Gam_rate(i) + ...
                0.5 * ( (ehmm.state(k).W.Mu_W(index,indr) .* ...
                sigmaprior ) * ...
                ehmm.state(k).W.Mu_W(index,indr)' + sum( ...
                sigmaprior .* ...
                ehmm.state(k).W.S_W(indr,index,index)'));
        end
        ehmm.state(k).alpha.Gam_shape = ehmm.state(k).alpha.Gam_shape + 0.5 * sum(indr);
    end
end
end