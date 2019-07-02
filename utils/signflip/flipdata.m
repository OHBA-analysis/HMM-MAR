function data = flipdata(data,T,flips,outputfile,force)
% Flip the channels of X according to flips (no. trials x no.channels),
% which is the output of findflip. If data is a collection of files, then
% it will write the output on disk.
%
% data          observations; either a struct with X (time series) and C (classes, optional),
%                             or a matrix containing the time series,
%                             or a list of file names
% T             length of series
% flips         The optimal flipping according to findflip.m
% outputfile    Cell with output file strings
% force         Flag to avoid the question of overwritting (only applies if
%               data is a collection of files and outputfile is not
%               specified
%
% Author: Diego Vidaurre, University of Oxford.

if nargin<5, force = 0; end 
if nargin<4, outputfile = []; end 

N = length(T); 

if iscell(data)  
    if isempty(outputfile) && ischar(data{1}) && ~force % check with user
        while true
            query = 'The result will overwrite the files, are you sure you want to continue? (0/1) ';
            answer = input(query);
            if answer==1 || answer==0, break; end
            disp('Please respond 0 or 1.')
        end
        if answer==0
            disp('Exiting then...')
            return;
        end
    end
    if ~ischar(data{1})
        for j = 1:N
            data{j} = flipdata_subject(data{j},T{j},flips(j,:));
            if ~isempty(outputfile)
                fsub = outputfile{j};
                writefile_sub;
            end
        end
    else
        for j = 1:N
            fsub = data{j};
            loadfile_sub;
            X = flipdata_subject(X,T{j},flips(j,:));
            if ~isempty(outputfile)
                fsub = outputfile{j};
            end
            writefile_sub;
        end
    end
else
    ndim = size(data,2);
    for j = 1:N
        ind = (1:T(j)) + sum(T(1:j-1));
        for d = 1:ndim
            if flips(j,d)
                data(ind,d) = -data(ind,d);
            end
        end
    end
    if ~isempty(outputfile)
        warning('For outputfile to be used, data must be specified as a cell')
    end
end

end


function X = flipdata_subject(X,T,flips)
N = length(T); ndim = size(X,2);
for j = 1:N
    ind = (1:T(j)) + sum(T(1:j-1));
    for d = 1:ndim
        if flips(d)
            X(ind,d) = -X(ind,d);
        end
    end
end
end