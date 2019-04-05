function status = checkGamma(Gamma,T,train,subj)
% Check that the state time courses are well-defined (no NaNs)
% and that the HMM inference wasn't trapped in a trivial local minima

if nargin<4, subj = 0; end

if any(isnan(sum(Gamma,2)))
    t = find(isnan(sum(Gamma,2)),1);
    if train.order>0
        d = train.order;
        Td = T-d;
    elseif length(train.embeddedlags)>1
        d = length(train.embeddedlags)-1;
        Td = T;
    else
        d = 0;
        Td = T;
    end
    cT = cumsum(Td); 
    i = find(cT>t,1);
    t = t + i*d;   
    if length(train.embeddedlags)>1, t = t - max(train.embeddedlags); end
    if subj > 0
        str = [' in subject ' num2str(subj) ', '];
    else
        str = ', '; 
    end
    disp(['The state time courses have NaN in t=' num2str(t) str 'where t indexes time in the data.'])
    disp('This is probably due to an artifact or an event too extreme to model around that time point.')
    disp('Please check your data arount that time point, and remove or smooth this event')
	error('Issue of precision')
end

K = size(Gamma,2);
status = (all(max(Gamma)<0.6) && all(min(Gamma)>(1/K/2)));

end
