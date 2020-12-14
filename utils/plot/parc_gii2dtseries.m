function parc_gii2dtseries(input,output,wbdir)
% Maps a parcellation from surface in func.gii format into dtseries format
% that can be used by makeMap
%
% Arguments
% input: cell with two elements, the first for the left hemisphere of the
%   parcellation in func.gii format, and the second for the right hemisphere.
%   E.g. input{1} = "DATA/MyConnectome/all_selected_L_new_parcel_renumbered.func.gii";
% output: single string where the CIFTI dtseries will be written
%   E.g. output = 'parcellation_all.dtseries.nii'
% wbdir: HCP workbench directory
%
% Note: HCPpipelines needs to be on the path 

if length(input)==3
    warning('Third element of input with subcortical data will be ignored');
end

tmp = [tempname '.dtseries.nii'];

what = system(sprintf('wb_command -cifti-create-dense-timeseries %s -left-metric %s -right-metric %s',...
    tmp,input{1},input{2}));
if what~=0
    path1 = getenv('PATH');
    path1 = [path1 ':' wbdir];
    setenv('PATH', path1);
    what = system(sprintf('wb_command -cifti-create-dense-timeseries %s -left-metric %s -right-metric %s',...
        tmp,input{1},input{2}));
    if what~=0
       error('wb_command could not be called successfully, is wbdir correct?')
    end
end

CIFTI = ciftiopen(tmp,[wbdir '/wb_command']);
c1 = CIFTI.cdata; 
CIFTI.cdata = zeros(size(c1,1), length(unique(c1))-1); 
for j = 1:max(c1)
   ind = (c1 == j); CIFTI.cdata(ind,j) = 1; 
end

ciftisave(CIFTI,output,[wbdir '/wb_command']);

delete(tmp);  

end