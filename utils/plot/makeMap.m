function maps = makeMap(hmm,k,parcellation_file,maskfile,...
    onconnectivity,centermaps,scalemaps,outputfile,wbdir)
% Project HMM maps into brain space for visualisation 
%
% hmm: hmm struct as comes out of hmmmar, 
%   or (ndim x states) matrix of activation values (only if (onconnectivity==0)
%   or (ndim x ndim x states) FC matrices (only if (onconnectivity>0)
% k: which state or states to plot, e.g. 1:4. If left empty, then all of them
% parcellation_file is either a nifti or a cifti file, containing either a
%   parcellation or an ICA decomposition
% maskfile: if using NIFTI, this is the mask to be used (leave empty when using CIFTIs)
%   e.g. 'std_masks/MNI152_T1_2mm_brain'
% centermaps: whether to center the maps according to the across-map average
%       (default: 0, unless there's no mean activity to show)
% scalemaps: whether to scale the maps so that each voxel has variance
%       equal 1 across maps; (default: 0)
% onconnectivity: whether or not to make the map using connectivity (>0) or mean activity (0). 
%   If onconnectivity==2, then the first eigenvector of the covariance matrix will be used
%   If onconnectivity==20, then the first eigenvector of the correlation matrix will be used. 
%   If onconnectivity==1, then the degree of the covariance matrix will be used. 
%   If onconnectivity==10, then the degree of the correlation matrix will be used. 
%   If onconnectivity==0 and there is no mean activity parameter, then the variance will be used.
%   If there's no mean parameter, it's 1 by default; 
%   if there's a mean parameter, then it's 0 by default. 
%   NOTE: the sign of an eigendecomposition is arbitrary. That means that when 
%       onconnectivity==2 or 20, positive areas of the map cannot be
%       interpreted as "more" or negative as "less"
%   NOTE2: when computing the degree (onconnectivity==1 or 10), the negative values of the 
%       covariance matrix are not considered (i.e. they are put to zero) 
% outputfile: where to put things (do not indicate extension)
%   e.g. 'my_directory/maps'
% wbdir: HCP workbench directory (if working with CIFTI,
%     or want to create a CIFTI from the volumetric file; if not, provide empty)
%   e.g. '/Applications/workbench/bin_macosx64'
%
% OUTPUT
% maps: (voxels by states) matrix with the projected maps
%
% Notes:
% if parcellation is a cifti, need to have HCPpipelines/global in path
% if parcellation is a nifti, need to have OSL in path
%
% Diego Vidaurre (2020)

if nargin < 6 || isempty(centermaps), centermaps = 0; end
if nargin < 7 || isempty(scalemaps), scalemaps = 0; end
if nargin < 8 || isempty(outputfile)
    outputfile = './map';
    warning('No output file specified, using "./map"')
end
if nargin < 9, wbdir = ''; end

if isstruct(parcellation_file) % func.gii's parcellation

elseif strcmp(parcellation_file(end-11:end),'dtseries.nii')
    CIFTI = ciftiopen(parcellation_file,[wbdir '/wb_command']);
    spatialMap = CIFTI.cdata; % vertices x components/parcels
elseif strcmp(parcellation_file(end-7:end),'func.gii')
    error('Needs parcellation in dtseries.nii format, run parc_gii2dtseries...')
elseif strcmp(parcellation_file(end-5:end),'nii.gz')
    NIFTI = parcellation(parcellation_file);
    spatialMap = NIFTI.to_matrix(NIFTI.weight_mask); % voxels x components/parcels
else
    error('Incorrect format: parcellation must have dtseries.nii or nii.gz extension')
end

q = size(spatialMap,2); 
if isstruct(hmm), K = length(hmm.state); 
elseif length(size(hmm))==3, K = size(hmm,3);
else, K = size(hmm,2);
end
if ~isempty(k), index_k = k; else, index_k = 1:K; end
% compensate the parcels to have comparable weights 

for j = 1:q % iterate through regions : make max value to be 1
    spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
end

do_HMM_pca = isstruct(hmm) && strcmpi(hmm.train.covtype,'pca'); 
if isstruct(hmm)
    is_there_mean = isfield(hmm.state(1),'W') && ~isempty(hmm.state(1).W.Mu_W);
else
    is_there_mean = (length(size(hmm))==2);
end
if nargin < 5 || isempty(onconnectivity)
    if ~do_HMM_pca && is_there_mean
        disp('onconnectivity option not specified, setting onconnectivity = 0..') 
        onconnectivity = 0;
    else
        disp('onconnectivity option not specified, setting onconnectivity = 1..') 
        onconnectivity = 1;
    end
elseif ~isstruct(hmm) % onconnectivity was specified
    if length(size(hmm))==3 && ~onconnectivity
        error('Incorrect format for hmm given onconnectivity==0')
    end
    if length(size(hmm))==2 && onconnectivity
        error('Incorrect format for hmm given onconnectivity==1')
    end
end
    
try
    for k = 1:K
        if isstruct(hmm) && (do_HMM_pca || ~is_there_mean) 
            % hmm is not an array or matrix
            C = getFuncConn(hmm,k,1);
            if k==1, mapsParc = zeros(size(C,1),K); end
            if onconnectivity>0
                mapsParc(:,k) = connmap(C,onconnectivity);
            else
                mapsParc(:,k) = diag(C);
            end
        elseif ~onconnectivity % always is_there_mean
            if isstruct(hmm)
                m = getMean(hmm,k,1);
            else
                m = hmm(:,k);
            end
            if k==1, mapsParc = zeros(length(m),K); end
            mapsParc(:,k) = m;
        else % onconnectivity, 
            if isstruct(hmm)
                C = getFuncConn(hmm,k,1);
            else
                C = hmm(:,:,k);            
            end
            if k==1, mapsParc = zeros(size(C,1),K); end
            mapsParc(:,k) = connmap(C,onconnectivity);
        end
    end
catch
    error('Incorrect HMM structure for that option of onconnectivity')
end

if q < size(mapsParc,1)
    warning('Parcellation has fewer regions than the HMM model - discarding the last ones')
    mapsParc = mapsParc(1:q,:);
end
    
maps = spatialMap * mapsParc; % voxels x states
if centermaps
    maps = maps - repmat(mean(maps,2),1,K);
end
if scalemaps
    maps = maps ./ repmat(std(maps,[],2),1,K);
end

maps = maps(:,index_k);

if strcmp(parcellation_file(end-11:end),'dtseries.nii')
    CIFTI.cdata = maps;
    ciftisave(CIFTI,[outputfile '.dtseries.nii'],[wbdir '/wb_command']);
else
    [mask,res,xform] = nii.load(maskfile);
    [directory,~] = fileparts(outputfile);
    nii.save(matrix2vols(maps,mask),res,xform,outputfile);
    path1 = getenv('PATH');
    if ~isempty(wbdir)
        attempt = 0; success = 0;
        while ~success && attempt<2
            try
                if ~isempty(directory)
                    osl_render4D([outputfile,'.nii.gz'],'savedir',directory,...
                        'interptype','trilinear','visualise',false)
                else
                    osl_render4D([outputfile,'.nii.gz'],...
                        'interptype','trilinear','visualise',false)
                end
                success = 1;
            catch
                if attempt == 0
                    path1 = [path1 ':' wbdir];
                    setenv('PATH', path1);
                    disp('.. adding workbench to the path')
                end
                attempt = attempt + 1;
                if attempt == 2
                    warning('Something failed in creating the CIFTI file: skipped')
                end
            end
        end
    end
end

end


function map = connmap(C,mode)
if mode > 5
    C = corrcov(C,0);
    mode = mode / 2; 
end
if mode == 2
    [V,D] = eig(C); d = diag(D); [~,j] = max(d); map = V(:,j);
else
    C(C<0) = 0;
    C(eye(size(C,1))==1) = 0; map = sum(C);
end
end
