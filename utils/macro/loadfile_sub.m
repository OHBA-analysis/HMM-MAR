if ~isempty(strfind(fsub,'.mat'))
    load(fsub,'X');
elseif ~isempty(strfind(fsub,'.txt'))
    X = dlmread(fsub);
else
    try
        D = spm_eeg_load(fsub);
        X = permute(D(:,:,:),[2 3 1]); clear D
        X = reshape(X,[size(X,1)*size(X,2) size(X,3)]);
    catch
        if ~exist('spm_eeg_load','file')
            error('spm_eeg_load() function not found - is SPM in the path?')
        else
            error('Incorrect data format - input files must be .mat, .txt or an SPM file');
        end
    end
end