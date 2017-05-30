% if data is a cell, transform into matrix format
if iscell(data)
    if size(data,1)==1, data = data'; end
    if iscellstr(data) % if comes in file name format
        dfilenames = data; t0 = 0;
        for i=1:length(dfilenames)
            if ~isempty(strfind(dfilenames{i},'.mat'))
                load(dfilenames{i},'X');
            elseif ~isempty(strfind(dfilenames{i},'.txt'))
                X = dlmread(dfilenames{i});
            else
                try
                    D = spm_eeg_load(dfilenames{i});
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
            if i==1, data = zeros(sum(T),size(X,2)); end
            try data(t0 + (1:size(X,1)),:) = X; t0 = t0 + size(X,1); 
            catch, error('The dimension of data does not correspond to T'); 
            end
        end
        if t0~=sum(T), error('The dimension of data does not correspond to T'); end
    else
        try data = cell2mat(data);
        catch, error('Subjects do not have the same number of channels');
        end
    end
end
