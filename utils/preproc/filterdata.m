function data = filterdata(data,T,Fs,freqbands)
% Filters the data using a Butterworth filter
% Author: Diego Vidaurre (2017)

Fn = Fs/2;  % Nyquist frequency
filterorder = 6;
[B, A] = buttfilter(filterorder,freqbands,Fn);

for n = 1:length(T)
    fo = filterorder; Bn = B; An = A; 
    ind = sum(T(1:n-1))+ (1:T(n));
    while 1
        try
            if isstruct(data)
                data.X(ind,:) = filtfilt(Bn, An, data.X(ind,:));
            else
                data(ind,:) = filtfilt(Bn, An, data(ind,:));
            end
            break
        catch
            fo = fo - 1;
            if fo<1, error('Filtering error, wrong dimensions?'); end
            [Bn, An] = buttfilter(fo,freqbands,Fn);
            continue;
        end
    end
end
    
end


function [B,A] = buttfilter(N, f, Fn)
if f(1)==0
    [B,A] = butter(N,f(2)/Fn,'low');
elseif isinf(f(2))
    [B,A] = butter(N,f(1)/Fn,'high');
else
    [B,A] = butter(N,[f(1)/Fn f(2)/Fn]);
end
end