function [X,T] = read_spm_file(spm_file,mat_file)
% Load an (continuous data) SPM file with filename specified in the spm_file  
% input parameter, and put the data into X in matrix format ready to be used by 
% the HMM. If mat_file is specified, X is saved in the file mat_file 
% Importantly, if there are bad time points in the time series, it removes
% them and returns a vector of segment lengths (T) for the HMM to take
% into account that the time series is not seamlessly continuous anymore. 
% In this case, although the hmmmar function can receive SPM files
% directly, it is strongly recommended to run this function for each
% subject before running the HMM, concatenate T across subjects,
% and then pass the mat_file's to the hmmmar function.
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

D = spm_eeg_load(spm_file);
X = D(:,:,:);
X = reshape(X,[D.nchannels,D.nsamples*D.ntrials]);

% select only good data
goodsamples = good_samples(D,[],[],[],0);
goodsamples = reshape(goodsamples,1,D.nsamples*D.ntrials);
X = X(:,goodsamples)';
options = struct(); 
T = cell2mat(getStateLifeTimes (goodsamples',length(goodsamples),options,1));

if min(T)<20
    warning('There are time segments with less than 20 time points')
end

if nargin > 1
   save(mat_file,'X','T') 
end

end
