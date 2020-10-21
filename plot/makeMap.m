function maps = makeMap(hmm,k,parcellation_file,maskfile,...
    centermaps,scalemaps,outputfile,wbdir)
% Project HMM maps into brain space for visualisation 
%
% hmm: hmm struct as comes out of hmmmar
% k: which state or states to plot, e.g. 1:4. If left empty, then all of them
% parcellation_file is either a nifti or a cifti file, containing either a
%   parcellation or an ICA decomposition
% maskfile: if using NIFTI, this is the mask to be used (leave empty when using CIFTIs)
%   e.g. 'std_masks/MNI152_T1_2mm_brain'
% centermaps: whether to center the maps according to the across-map average
%       (default: 0)
% scalemaps: whether to scale the maps so that each voxel has variance
%       equal 1 across maps; (default: 0)
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

if nargin < 5 || isempty(centermaps), centermaps = 0; end
if nargin < 6 || isempty(scalemaps), scalemaps = 0; end
if nargin < 7 || isempty(outputfile)
    outputfile = './map';
    warning('No output file specified, using "./map"')
end
if nargin < 8, wbdir = ''; end

if strcmp(parcellation_file(end-11:end),'dtseries.nii')
    CIFTI = ciftiopen(parcellation_file,[wbdir '/wb_command']);
    spatialMap = CIFTI.cdata; % vertices x components/parcels
elseif strcmp(parcellation_file(end-5:end),'nii.gz')
    NIFTI = parcellation(parcellation_file);
    spatialMap = NIFTI.to_matrix(NIFTI.weight_mask); % voxels x components/parcels
else
    error('Incorrect format: parcellation must have dtseries.nii or nii.gz extension')
end

q = size(spatialMap,2); K = length(hmm.state);
if ~isempty(k), index_k = k; else, index_k = 1:K; end
% compensate the parcels to have comparable weights 

for j = 1:q % iterate through regions : make max value to be 1
    spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
end

do_HMM_pca = strcmpi(hmm.train.covtype,'pca');
try
    mapsParc = zeros(q,K);
    is_there_mean = isfield(hmm.state(1),'W') && ~isempty(hmm.state(1).W.Mu_W);
    if do_HMM_pca
        disp('PCA states: using variance')
    elseif ~is_there_mean && ~do_HMM_pca
        disp('No mean was used in the estimation (options.zeromean=1); using variance instead')
    end
    for k = 1:K
        if do_HMM_pca % here we could also do sum(hmm.state(k).W.Mu_W.^2,2)
            C = getFuncConn(hmm,k,1);
            mapsParc(:,k) = diag(C);
        elseif is_there_mean
            mapsParc(:,k) = getMean(hmm,k,1);
        else
            C = getFuncConn(hmm,k,1);
            mapsParc(:,k) = diag(C);
        end
    end
catch
    error('Incorrect HMM structure')
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
