function [order,exptimelag] = higherorder(minHz,Fs,L,orderoffset)
% tells the order and exponential time lag for covering up to the slowest
% frequency, using L lags

INC = 1.05;
order = ceil(Fs / minHz); exptimelag = 1.01;
if nargin<4, orderoffset = 0; end

if nargin>=3
    
while 1
   orders = formorders(order,orderoffset,0,exptimelag); 
   if length(orders)>L
       exptimelag = exptimelag * INC;
   elseif length(orders)<L
       INC = 0.99;
       exptimelag = exptimelag * INC;
   else
       break
   end
end

end