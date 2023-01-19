function graphs = makeSpectralGraph(sp_fit,k,freqindex,type,parcellation_file,...
    maskfile,centergraphs,scalegraphs,threshold,outputfile,figure_number)
% Project spectral connectomes into brain space for visualisation  
%
% sp_fit: spectral fit, from hmmspectramt, hmmspectramar, spectbands or spectdecompose
% k: which state or states to plot, e.g. 1:4. If left empty, then all of them
% freqindex: which frequency bin or band to plot; it can be a value or a
%   range, in which case it will sum across the range.
%   e.g. 1:20, to integrate the first 20 frequency bins, or, if sp_fit is
%   already organised into bands: 2 to plot the second band.
% type: which connectivity measure to use: 1 for coherence, 2 for partial coherence
% parcellation_file is either a nifti or a cifti file, containing either a
%   parcellation or an ICA decomposition
% maskfile: mask to be used with the right spatial resolution
%   e.g. 'std_masks/MNI152_T1_2mm_brain'
% centergraphs: whether to center the graphs according to the across-map average
%       (default: 0)
% scalegraphs: whether to scale the graphs so that each voxel has variance
%       equal 1 across maps; (default: 0)
% threshold: proportion threshold above which graph connections are
%       displayed (between 0 and 1, the higher the fewer displayed connections)
% outputfile: where to put things (do not indicate extension)
%   e.g. 'my_directory/maps'
% maskfile: if using NIFTI, mask to be used 
%   e.g. 'std_masks/MNI152_T1_2mm_brain'
% figure_number: number of figure to display
%
% OUTPUT:
% graph: (voxels by voxels by state) array with the estimated connectivity maps
%
% Notes: need to have OSL in path
%
% Diego Vidaurre (2020)

if nargin < 4 || isempty(type), disp('type not specified, using coherence.'); type = 1; end

if nargin < 7 || isempty(centergraphs), centergraphs = 0; end
if nargin < 8 || isempty(scalegraphs), scalegraphs = 0; end
if nargin < 9 || isempty(threshold), threshold = 0.95; end
if nargin < 10 || isempty(outputfile), outputfile = []; end
if nargin < 11 || isempty(figure_number), figure_number = randi(100); end

if ~isfield(sp_fit.state(1),'psd')
    error('Input must be a spectral estimation, e.g. from hmmspectramt')
end

try
    NIFTI = parcellation(parcellation_file);
    spatialMap = NIFTI.to_matrix(NIFTI.weight_mask); % voxels x components/parcels
catch
    error('Incorrect format: parcellation must have nii.gz extension')
end

try
    mni_coords = find_ROI_centres_2(spatialMap, maskfile, 0); % adapted from OSL
catch
    error('Error with OSL: find_ROI_centres in path?')
end   

ndim = size(spatialMap,2); K = length(sp_fit.state);
if ~isempty(k), index_k = k; else, index_k = 1:K; end

% compensate the parcels to have comparable weights 
for j = 1:ndim % iterate through regions : make max value to be 1
    spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
end

graphs = zeros(ndim,ndim,K);
edgeLims = [4 8]; colorLims = [0.1 1.1]; sphereCols = repmat([30 144 255]/255, ndim, 1);

for k = 1:K
    if type==1
        C = permute(sum(sp_fit.state(k).coh(freqindex,:,:),1),[2 3 1]);
    elseif type==2
        try
            C = permute(sum(sp_fit.state(k).pcoh(freqindex,:,:),1),[2 3 1]);
        catch
            error('Partial coherence was not estimated so it could not be used')
        end
    else
        error('Not a valid value for type')
    end
    C(eye(ndim)==1) = 0;
    graphs(:,:,k) = C;
end

if centergraphs
    graphs = graphs - repmat(mean(graphs,3),[1 1 K]);
end
if scalegraphs
    graphs = graphs ./ repmat(std(graphs,[],3),[1 1 K]);
end

for ik = 1:length(index_k)
    k = index_k(ik); 
    C = graphs(:,:,k);
    %c = C(triu(true(ndim),1)==1); c = sort(c); c = c(end-1:-1:1); 
    %th = c(round(length(c)*(1-threshold))); 
    %C(C<th) = NaN; 
    figure(k+figure_number);
    osl_braingraph(C, colorLims, repmat(0.5,ndim,1), [0.1 1.1], mni_coords, ...
        [], 100*threshold, sphereCols, edgeLims);
    colorbar off
    if ~isempty(outputfile)
        saveas(gcf,[outputfile '_' num2str(k) '.png'])
    end
    %fig_handle = gcf;
end

graphs = graphs(:,:,index_k);

end


function coords = find_ROI_centres_2(spatialMap, brainMaskName, isBinary)
% based on the OSL's find_ROI_centres
[nVoxels, nParcels] = size(spatialMap);
MNIcoords     = osl_mnimask2mnicoords(brainMaskName);
assert(ROInets.rows(MNIcoords) == nVoxels);
for iParcel = nParcels:-1:1
    map = spatialMap(:, iParcel);
    % find ROI
    if isBinary
        cutOff = 0;
    else
        % extract top 5% of values
        cutOff = prctile(map, 95);
    end%if
    ROIinds = (map > cutOff);
    % find weightings
    if isBinary
        masses = ones(sum(ROIinds), 1);
    else
        masses = map(ROIinds);
    end
    % find CoM
    CoM = (masses' * MNIcoords(ROIinds,:)) ./ sum(masses);
    coords(iParcel, :) = CoM;
end
end

