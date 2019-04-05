function s = getAccuracy(X,Y,T,Gamma,options,do_preproc)
% Returns a time resolved (cross-validated) measure of accuracy
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

if nargin<6, do_preproc = 1; end
if nargin<5, options = struct(); end
if ~all(T(1)==T)
    error('Synchronisity can only be measured if trials have the same length'); 
end
if sum(T) ~= size(X,1)
    T = T - 1;
end
if sum(T) ~= size(X,1)
    error('Dimensions of data are not consistent with T')
end

options.mode = 1;
options.CVmethod = 1; 
options.Nperm = 1; 

s = tudacv(X,Y,T,options,Gamma,do_preproc);

end