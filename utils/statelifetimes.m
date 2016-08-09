function [tt, states] = statelifetimes(Gamma,T,orderstates,Hz)
% tt is the time that the system stays in this state
% classes refers to as which state tt corresponds to
% this function was before called 'collect_times.m'
%
% Author: Diego Vidaurre ,University of Oxford, 2016

if nargin<2, T = size(Gamma,1); end
if nargin<3, orderstates = 0; end
if nargin<4, Hz = 1; end

tt = [];
states = [];

if size(Gamma,2)==1
   vp = Gamma; K = max(vp);
   Gamma = zeros(size(vp,1),K);
   for k=1:K, Gamma(vp==k,k) = 1; end
end

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
        active = 0;
        for t=1:T(n)
            if (Gamma_n(t,k)==0 || t==T(n)) && active==1 % end or change of the activation
                if t==T(n) && Gamma_n(t,k)==1
                    count = count + 1;
                end;
                tt = [tt count];
                states = [states k];
                count = 0;
            end
            if Gamma_n(t,k)==1 && active==0 % we start a new activation
                count = 1;
            end;
            if Gamma_n(t,k)==1 && active==1 % we keep in this activation
                count = count + 1;
            end
            active = Gamma_n(t,k);
        end
    end  
end

tt = tt / Hz;

end


        
    


