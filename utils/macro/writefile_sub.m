if ~isempty(strfind(fsub,'.mat'))
    save(fsub,'X','-append')
elseif ~isempty(strfind(fsub,'.txt'))
    dlmwrite(fsub,X);
else % no extension - assumed to be an SPM file
    error('For now, we can only deal with .mat and .txt files')
end