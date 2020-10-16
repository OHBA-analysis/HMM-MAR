function graphs = makeSpectralConnectivityCircle(sp_fit,freqindex,type,...
    labels,centergraphs,scalegraphs,threshold)
% Plot spectral connectomes in connectivity circle format
%
% sp_fit: spectral fit, from hmmspectramt, hmmspectramar, spectbands or spectdecompose
% freqindex: which frequency bin or band to plot; it can be a value or a
%   range, in which case it will sum across the range.
%   e.g. 1:20, to integrate the first 20 frequency bins, or, if sp_fit is
%   already organised into bands: 2 to plot the second band.
% type: which connectivity measure to use: 1 for coherence, 2 for partial coherence
% labels : names of the regions
% centermaps: whether to center the maps according to the across-map average
% scalemaps: whether to scale the maps so that each voxel has variance
%       equal 1 across maps
% threshold: proportion threshold above which graph connections are
%       displayed (between 0 and 1, the higher the fewer displayed connections)
%
% OUTPUT:
% graph: (regions by regions by state) array with the estimated connectivity maps
%
% Notes:
% It needs the circularGraph toolbox from Matlab in the path: 
%   https://github.com/paul-kassebaum-mathworks/circularGraph
%
% Diego Vidaurre (2020)

if isempty(type), disp('type not specified, using coherence.'); type = 1; end

if nargin < 5 || isempty(centergraphs), centergraphs = 0; end
if nargin < 6 || isempty(scalegraphs), scalegraphs = 0; end
if nargin < 7 || isempty(threshold), threshold = 0.95; end

if ~isfield(sp_fit.state(1),'psd')
    error('Input must be a spectral estimation, e.g. from hmmspectramt')
end

K = length(sp_fit.state); ndim = size(sp_fit.state(1).psd,2);

if nargin < 4 || isempty(labels)
    labels = cell(ndim,1);
    for j = 1:ndim
       labels{j} = ['Parcel ' num2str(j)];
    end
end

graphs = zeros(ndim,ndim,K);

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

for k = 1:K
    C = graphs(:,:,k);
    c = C(triu(true(ndim),1)==1); c = sort(c); c = c(end-1:-1:1);
    th = c(round(length(c)*(1-threshold)));
    C(C<th) = 0;
    try
        figure(k+200)
        circularGraph(C,'Label',labels);
    catch
        error('Please get the Matlab circularGraph toolbox first')
    end
end

% end
% 
% 
% 
% function nets_netweb_2(netF,netP,sumpics,outputdir)
% % borrowed from Steve's fslnets
% 
% % replicate functionality from nets_hierarchy
% grot=prctile(abs(netF(:)),99); netmatL=netF/grot; netmatH=netP/grot;
% usenet=netmatL;  usenet(usenet<0)=0;
% N=size(netmatL,1);  grot=prctile(abs(usenet(:)),99); usenet=max(min(usenet/grot,1),-1)/2;
% DD = 1:N;
% for J = 1:N, for I = 1:J-1,   y((I-1)*(N-I/2)+J-I) = 0.5 - usenet(I,J);  end; end;
% linkages=linkage(y,'ward');
% set(0,'DefaultFigureVisible','off');
% figure;[~,~,hier]=dendrogram(linkages,0,'colorthreshold',0.75);
% close;set(0,'DefaultFigureVisible','on');
% clusters=cluster(linkages,'maxclust',10)';
% 
% mkdir(outputdir)
% netjs = [fileparts(which('makeConnectivityCircle')) '/netjs'];
% copyfile(netjs,outputdir) % copy javascript stuff into place
% NP=sprintf('%s/data/dataset1',outputdir);
% save(sprintf('%s/Znet1.txt',NP),'netF','-ascii');
% save(sprintf('%s/Znet2.txt',NP),'netP','-ascii');
% save(sprintf('%s/hier.txt',NP),'hier','-ascii');
% save(sprintf('%s/linkages.txt',NP),'linkages','-ascii');
% save(sprintf('%s/clusters.txt',NP),'clusters','-ascii');
% mkdir(sprintf('%s/melodic_IC_sum.sum',NP));
% for i=1:length(DD)
%   system(sprintf('/bin/cp %s.sum/%.4d.png %s/melodic_IC_sum.sum/%.4d.png',sumpics,DD(i)-1,NP,i-1));
% end
% 
% end
% 







