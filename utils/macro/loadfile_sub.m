if ~isempty(strfind(fsub,'.mat'))
    dat = load(fsub);
    if isfield(dat,'X')
        X = dat.X;
    else
        X = getfield(dat,char(fieldnames(dat)));
    end
    clear dat
elseif ~isempty(strfind(fsub,'.txt'))
    X = dlmread(fsub);
else % no extension - assumed to be an SPM file
    try % Note that we are not taking care of bad samples here
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
