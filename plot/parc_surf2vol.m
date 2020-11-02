function parc_surf2vol(input,surface,volume_space,output,wbdir)
% Maps a parcellation from surface into volumetric space, for example so
% that makeBrainGraph can be used with data that comes in surface space.
%
% Arguments
% input: cell with two elements, the first for the left hemisphere of the
%   parcellation in func.gii format, and the second for the right hemisphere.
%   E.g. input{1} = "DATA/MyConnectome/all_selected_L_new_parcel_renumbered.func.gii";
%   A third optional element can contain a matrix (X Y Z nregions)
%   containing the locations of subcortical regions of interest. 
% surface: cell with two elements, each with the surface to use coordinates from. 
%   The first element corresponds to the first hemisphere and the second to the
%   second hemisphere. 
%   E.g. surface{1} = "OSLDIR/std_masks/ParcellationPilot.L.midthickness.32k_fs_LR.surf.gii"
% volume_space: a volume file in the desired output volume space
%   E.g. volume_space = OSLDIR/std_masks/MNI152_T1_2mm_brain.nii.gz
% output: single string where the NIFTI parcellation will be written
%   E.g. output = 'parcellation_all.nii.gz'
% wbdir: HCP workbench directory

% Note: HCPpipelines needs to be on the path, and osl_startup must has been run

tmp = tempname;
tmp1 = [tmp '_L.func.gii']; 
tmp2 = [tmp '_R.func.gii'];
tmp1_nii = [tmp '_L.nii.gz']; 
tmp2_nii = [tmp '_R.nii.gz']; 

mm_nearestneigh = '10'; % I have not explored the effect of this parameter 

g1 = gifti(input{1}); c1 = g1.cdata;
g2 = gifti(input{2}); c2 = g2.cdata;

g1.cdata = zeros(size(g1.cdata,1), length(unique(c1))+length(unique(c2))-2); 
g2.cdata = zeros(size(g2.cdata,1), length(unique(c1))+length(unique(c2))-2); 

for j = 1:max(c1)
   ind = (c1 == j); g1.cdata(ind,j) = 1; 
end
for j = max(c1)+1:max(c2)
   ind = (c2 == j); g2.cdata(ind,j) = 1; 
end

save(g1,tmp1)
save(g2,tmp2)

what = system(sprintf('wb_command -metric-to-volume-mapping %s %s %s %s -nearest-vertex %s',...
    tmp1,surface{1},volume_space,tmp1_nii,mm_nearestneigh));
if what~=0
    path1 = getenv('PATH');
    path1 = [path1 ':' wbdir];
    setenv('PATH', path1);
    what = system(sprintf('wb_command -metric-to-volume-mapping %s %s %s %s -nearest-vertex %s',...
        tmp1,surface{1},volume_space,tmp1_nii,mm_nearestneigh));
    if what~=0
       error('wb_command could not be called successfully, is wbdir correct?')
    end
end
    
system(sprintf('wb_command -metric-to-volume-mapping %s %s %s %s -nearest-vertex %s',...
    tmp2,surface{2},volume_space,tmp2_nii,mm_nearestneigh));

[~,res,xform] = nii.load('~/Work/Matlab/ohba_analysis/std_masks/MNI152_T1_2mm_brain.nii.gz');

D1 = nii.load(tmp1_nii);
D2 = nii.load(tmp2_nii);
D = D1 + D2; % size (91,109,91,nregions); 
if length(input)==3
    D = cat(4,D,input{3});
end

nii.save(D,res,xform,output);

delete(tmp1); delete(tmp2); delete(tmp1_nii); delete(tmp2_nii);

end