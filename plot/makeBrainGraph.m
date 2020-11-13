function graphs = makeBrainGraph(hmm,k,parcellation_file,maskfile,...
    centergraphs,scalegraphs,partialcorr,threshold,outputfile,figure_number)
% Project HMM connectomes into brain space for visualisation  
%
% hmm: hmm struct as comes out of hmmmar
% k: which state or states to plot, e.g. 1:4. If left empty, then all of them
% parcellation_file is either a nifti or a cifti file, containing either a
%   parcellation or an ICA decomposition
% maskfile: mask to be used with the right spatial resolution
%   e.g. 'std_masks/MNI152_T1_2mm_brain'
% centergraphs: whether to center the graphs according to the across-map average
%       (default: 0)
% scalegraphs: whether to scale the graphs so that each voxel has variance
%       equal 1 across maps; (default: 0)
% partialcorr: whether to use a partial correlation matrix or a correlation
%   matrix (default: 0)
% threshold: proportion threshold above which graph connections are
%       displayed (between 0 and 1, the higher the fewer displayed connections)
% outputfile: where to put things (do not indicate extension)
%   e.g. 'my_directory/maps'
% figure_number: number of figure to display
%
% OUTPUT:
% graph: (voxels by voxels by state) array with the estimated connectivity maps
%
% Notes: need to have OSL in path
%
% Diego Vidaurre (2020)

if nargin < 5 || isempty(centergraphs), centergraphs = 0; end
if nargin < 6 || isempty(scalegraphs), scalegraphs = 0; end
if nargin < 7 || isempty(partialcorr), partialcorr = 0; end
if nargin < 8 || isempty(threshold), threshold = 0.95; end
if nargin < 9 || isempty(outputfile), outputfile = []; end
if nargin < 10 || isempty(figure_number), figure_number = randi(100); end

do_HMM_pca = strcmpi(hmm.train.covtype,'pca');
if ~do_HMM_pca && ~strcmp(hmm.train.covtype,'full')
    error('Cannot great a brain graph because the states do not contain any functional connectivity')
end

K = length(hmm.state); if ~isempty(k), index_k = k; else, index_k = 1:K; end

for k = 1:K
    if partialcorr
        [~,~,~,C] = getFuncConn(hmm,k,1);
    else
        [~,C] = getFuncConn(hmm,k,1);
    end
    if k == 1
       ndim = size(C,1);
       graphs = zeros(ndim,ndim,K);
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

if strcmp(parcellation_file(end-11:end),'dtseries.nii')
    disp('Cannot make a brain graph on surface space.')
    disp('Map the parcellation to volumetric space first using parc_surf2vol()')
    return
elseif strcmp(parcellation_file(end-7:end),'func.gii')
    disp('Cannot make a brain graph on surface space.')
    disp('Map the parcellation to dtseries using parc_gii2dtseries,')
    disp('then to volumetric space using parc_surf2vol()')
    return
elseif ~strcmp(parcellation_file(end-5:end),'nii.gz')
    error('Incorrect format: parcellation must have nii.gz extension')
end

NIFTI = parcellation(parcellation_file);
spatialMap = NIFTI.to_matrix(NIFTI.weight_mask); % voxels x components/parcels

% compensate the parcels to have comparable weights 
for j = 1:ndim % iterate through regions : make max value to be 1
    spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
end

try
    mni_coords = find_ROI_centres_2(spatialMap, maskfile, 0); % adapted from OSL
catch
    error('Error finding ROI centres: wrong mask?')
end

edgeLims = [4 8]; 
sphereCols = repmat([30 144 255]/255, ndim, 1);

for ik = 1:length(index_k)
    k = index_k(ik); 
    C = graphs(:,:,k);
    %c = C(triu(true(ndim),1)==1); c = sort(c); c = c(end-1:-1:1); 
    %th = c(round(length(c)*(1-threshold))); 
    %C(C<th) = NaN; 
    figure(k+figure_number); m = max(abs(C(:)));
    colorLims = [-0.9*m 0.9*m];
    osl_braingraph(C, colorLims, repmat(0.2,ndim,1), [0.1 1.1], mni_coords, ...
        [], 100*threshold, sphereCols, edgeLims);
    %colorbar off
    if ~isempty(outputfile)
        view(0,80);
        saveas(gcf,[outputfile '_' num2str(k) '_TOP.png'])
        view(60,-20);
        saveas(gcf,[outputfile '_' num2str(k) '_RIGHT.png'])
        view(-60,-20);
        saveas(gcf,[outputfile '_' num2str(k) '_LEFT.png'])
        view(0,0);
        saveas(gcf,[outputfile '_' num2str(k) '_BACK.png'])        
        view(180,0);
        saveas(gcf,[outputfile '_' num2str(k) '_FRONT.png'])        
        close(k+100)
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



