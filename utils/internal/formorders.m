function [orders,order] = formorders(order,orderoffset,timelag,exptimelag)
%
% get vector of lags to look for the MAR
% 
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<2, orderoffset = 0; end
if nargin<3, timelag = 1; end
if nargin<4, exptimelag = 1; end

if order > 0
    if exptimelag>1
        orders = orderoffset+1;
        sep = 0; 
        while 1
            neworder = orders(end) + round(exptimelag^sep);
            if neworder>order, break; end
            if neworder~=orders(end), orders = [orders neworder]; end
            sep = sep + 1; 
        end
    else
        orders = orderoffset + (1:timelag:(order-orderoffset)); % quicker
        % drop = find(orders>order); 
        % if ~isempty(drop)
        %     drop = drop(1);
        %     orders = orders(1:drop-1); 
        % end 
    end
    order = orders(end);
else
    orders = []; order = 0;
end

end
