if ~isempty(strfind(fsub,'.mat'))
    if exist(fsub,'file')
        save(fsub,'X','-append')
    else
        save(fsub,'X')
    end
elseif ~isempty(strfind(fsub,'.txt'))
    dlmwrite(fsub,X);
else % no extension - assumed to be an SPM file
    error('For now, we can only deal with .mat and .txt files')
end