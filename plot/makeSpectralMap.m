function maps = makeSpectralMap(sp_fit,k,freqindex,parcellation_file,maskfile,...
    centermaps,scalemaps,outputfile,wbdir)
% Make brain maps from spectral power in NIFTI and CIFTI formats  
%
% sp_fit: spectral fit, from hmmspectramt, hmmspectramar, spectbands or spectdecompose
% k: which state or states to plot, e.g. 1:4. If left empty, then all of them
% freqindex: which frequency bin or band to plot; it can be a value or a
%   range, in which case it will sum across the range.
%   e.g. 1:20, to integrate the first 20 frequency bins, or, if sp_fit is
%   already organised into bands: 2 to plot the second band.
%   By default, it sumes across all frequency bins or bands
% parcellation_file contains either a parcellation or an ICA decomposition
% maskfile: if using NIFTI, this is the mask to be used
%   e.g. 'std_masks/MNI152_T1_2mm_brain'
% centermaps: whether to center the maps according to the across-map average
%       (default: 0)
% scalemaps: whether to scale the maps so that each voxel has variance
%       equal 1 across maps; (default: 0)
% outputfile: where to put things (do not indicate extension)
%   e.g. 'my_directory/maps'
% wbdir: HCP workbench directory, if you want to create a CIFTI
%    from the volumetric file; if not, provide empty)
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

try
    NIFTI = parcellation(parcellation_file);
    spatialMap = NIFTI.to_matrix(NIFTI.weight_mask); % voxels x components/parcels
catch
    error('Incorrect format: parcellation must have nii.gz extension')
end

if ~isfield(sp_fit.state(1),'psd')
    error('Input must be a spectral estimation, e.g. from hmmspectramt')
end

q = size(spatialMap,2); K = length(sp_fit.state);
if ~isempty(k), index_k = k; else, index_k = 1:K; end
% compensate the parcels to have comparable weights 

for j = 1:q % iterate through regions : make max value to be 1
    spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
end

mapsParc = zeros(q,K);
for k = 1:K
    try
        if isempty(freqindex)
            disp('No frequency range specified, adding all them up.')
            freqindex = 1:size(sp_fit.state(k).psd,1);
        end
        mapsParc(:,k) = diag(permute(sum(sp_fit.state(k).psd(freqindex,:,:),1),[2 3 1]));
    catch
        error('Error indexing spectra: wrong freqindex?')
    end
end
maps = spatialMap * mapsParc; % voxels x states
if centermaps
    maps = maps - repmat(mean(maps,2),1,K);
end
if scalemaps
    maps = maps ./ repmat(std(maps,[],2),1,K);
end

maps = maps(:,index_k);

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













