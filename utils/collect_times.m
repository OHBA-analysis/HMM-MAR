function [tt, classes]=collect_times(Gamma,T,orderstates)
% tt is the number of time points that the system stays in this state
% classes refers to as which state tt corresponds to

if nargin<3, orderstates = 1; end

tt = [];
classes = [];

Gamma_bin = Gamma>0.5;
Gammasum = sum(Gamma_bin);
if orderstates==1
    [~,ord] = sort(Gammasum,'descend');
    Gamma_bin = Gamma_bin(:,ord);
end
%Gamma_bin = Gamma_bin(:,Gammasum>0);

K = size(Gamma_bin,2);
for n=1:length(T)
    Gamma_n = Gamma_bin(sum(T(1:n-1))+1:sum(T(1:n)),:);
    for k=1:K
        if sum(Gamma_n(:,k))==0, continue; end
        count = 0;
        state = 0;
        for t=1:T(n)
            if (Gamma_n(t,k)==0 || t==T(n)) && state==1 % end or change of the activation
                if t==T(n) && Gamma_n(t,k)==1
                    count = count + 1;
                end;
                tt = [tt count];
                classes = [classes k];
                count = 0;
            end
            if Gamma_n(t,k)==1 && state==0 % we start a new activation
                count = 1;
            end;
            if Gamma_n(t,k)==1 && state==1 % we keep in this activation
                count = count + 1;
            end
            state = Gamma_n(t,k);
        end
    end
end


        
    


