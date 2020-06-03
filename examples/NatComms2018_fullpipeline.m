% Code used in Vidaurre et al. (2018) Nature Communications
%
% Detailed documentation and further examples can be found in:
% https://github.com/OHBA-analysis/HMM-MAR
% This pipeline must be adapted to your particular configuration of files. 
%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP THE MATLAB PATHS AND FILE NAMES
% make sure that fieldtrip and spm are not in your matlab path

mydir = '/home/diegov/MATLAB'; % adapt to yours

workingdir = [mydir '/data/MRC_Notts/tests/'];
codedir = [mydir '/ohba_analysis/'];
hmmdir = [mydir '/HMM-MAR/']; % latest version on git
spmfilesdir = fullfile(workingdir,'spm/');
matrixfilesdir = fullfile(workingdir,'data/');
datadir = [workingdir '/raw_data/'];
outputdir = [workingdir '/hmm_out/'];
mapsdir = [workingdir '/maps/'];
osldir = [codedir '/osl-core'];

addpath(osldir)
osl_startup

addpath([codedir '/ohba-external/ohba_utils'])
addpath([codedir '/ohba-external/nifti_tools'])
addpath(genpath([codedir '/spm12']))

subjectdirs = dir([datadir '/3*']);
subjectdirs = sort_nat({subjectdirs.name});

ctf_files            = cell(size(subjectdirs));
spm_files            = cell(size(subjectdirs));
structural_files        = cell(size(subjectdirs));
pos_files               = cell(size(subjectdirs));

% Cycle through subjects (some don't have resting state, so we skip)
for s = 1:length(subjectdirs)
    
    dsfile = dir([datadir subjectdirs{s} '/*Eyes_Open*.ds']); 
    if ~isempty(dsfile)
        ctf_files{s} = [datadir subjectdirs{s} '/' dsfile.name];
        % set up a list of SPM MEEG object file names (we only have one here)
        spm_files{s}    = [spmfilesdir subjectdirs{s} 'opt_eo.mat'];
    end
    
    % structural files
    try
        niifile = [datadir subjectdirs{s} '/' subjectdirs{s} '_CRG.nii'];
        structural_files{s} = niifile;
    end
    
    % list of head position files
    pfname=[datadir 'pos_files/' subjectdirs{s} '.pos'];
    pf = dir(pfname); 
    if ~isempty(pf)
        pos_files{s}=pfname;
    end
    
end

subjects_to_do = find(~cellfun(@isempty,spm_files) & ...
    ~cellfun(@isempty,structural_files) & ...
    ~cellfun(@isempty,pos_files));

%%%%%%%%%%%%%%%%%%%%%%%%%
%% CTF data preprocessing
% CONVERT FROM .ds TO AN SPM MEEG OBJECT:
  
for ss=1:length(subjects_to_do) % iterates over subjects    
    s=subjects_to_do(ss);
    osl_import(ctf_files{s},'outfile',spm_files{s});
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sort out fiducials:
for ss=1:length(subjects_to_do)   
    f = subjects_to_do(ss);
    spm_file = prefix(spm_files{f},'');
    D = spm_eeg_load(spm_file);
    fID = fopen(pos_files{f});
    fid_data = textscan(fID, '%s %f %f %f');
    fclose(fID);
    fid_new = [];
    % Fiducials:
    fid_inds = [find(strcmpi(fid_data{1},'nasion')) ...
        find(strcmpi(fid_data{1},'left')) ...
        find(strcmpi(fid_data{1},'right'))];
    fid_new.fid.pnt = [fid_data{2}(fid_inds) fid_data{3}(fid_inds) fid_data{4}(fid_inds)] * 10;
    fid_new.fid.label = {'nas';'lpa';'rpa'};
    % Headshape:
    hs_inds = setdiff(2:length(fid_data{1}),fid_inds);
    fid_new.pnt = [fid_data{2}(hs_inds) fid_data{3}(hs_inds) fid_data{4}(hs_inds)] * 10;  
    fid_new.unit = 'mm';
    % Labels:
    fid_labels = fid_data{1};
    D = fiducials(D,fid_new);
    D.save;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%% OPT

session_name = 'eo';
preproc_name = 'new250';

opt = [];
opt.dirname = [workingdir preproc_name '_' session_name '.opt/'];

spm_files_in = cell(length(subjects_to_do),1);
structural_files_in = cell(length(subjects_to_do),1);

for ss=1:length(subjects_to_do)
    f = subjects_to_do(ss);
    spm_files_in{ss} = prefix(spm_files{f},'');
    structural_files_in{ss} = structural_files{f};
    % label artefact chans:
    D = spm_eeg_load(spm_files_in{ss});
    D = D.chantype(find(strcmp(D.chanlabels,'EEG060')),'EMG');
    D = D.chantype(find(strcmp(D.chanlabels,'EEG059')),'ECG');
    D = D.chantype(find(strcmp(D.chanlabels,'EEG057')),'EOG1');
    D = D.chantype(find(strcmp(D.chanlabels,'EEG058')),'EOG2');
    D.save;
end

opt.datatype='ctf';
opt.spm_files=spm_files_in;

opt.downsample.do = 1;
opt.downsample.freq = 250;

% HIGHPASS AND NOTCH FILTERING
opt.highpass.do = 1;
opt.highpass.cutoff = 0.1;
opt.mains.do = 1;  

% AFRICA DENOISING
opt.africa.do = 1;
opt.africa.todo.ica = 0; % set to 1 if you want to do ICA denoising
opt.africa.todo.ident = 0;
opt.africa.todo.remove = 0;
opt.africa.ident.func = @identify_artefactual_components_auto;
opt.africa.ident.kurtosis_wthresh = 0.2;
opt.africa.ident.max_num_artefact_comps = 2;
opt.africa.precompute_topos = 1;

opt.bad_segments.do = 1; 
%opt.bad_segments.wthresh_ev = [0.05 0.05];

opt.coreg.do = 1;
opt.coreg.mri = structural_files_in;
opt.coreg.use_rhino = 1;
opt.coreg.forward_meg = 'MEG Local Spheres';

% Epoching settings
opt.epoch.do = 0;
opt.outliers.do = 0;

% opt=opt_consolidate_results(opt);
opt = osl_run_opt(opt);

%%%%%%%%%%%%%%%%%%%%%%%%%
%% copy and rename spm files to have names session1...

session_name = 'eo';
spm_files_opt = cell(length(opt.results.spm_files),1);
for ss=1:length(opt.results.spm_files)
    S = [];
    S.D = opt.results.spm_files{ss};
    S.outfile = [spmfilesdir preproc_name '_' session_name '_session' num2str(ss)];
    Dnew = spm_eeg_copy(S);
    % write log of how original spm files are linked to copies
    spm_files_opt{ss} = S.outfile;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%% filter and beamforming

freq = [1 98];

session_name = 'eo';
preproc_name='new250';
dirname = [spmfilesdir preproc_name '_' session_name '/'];
bfdir = [dirname 'bfnew_' num2str(freq(1)) 'to' num2str(freq(2)) 'hz/'];
mkdir(dirname); mkdir(bfdir);
mni_coords = osl_mnimask2mnicoords(fullfile(codedir,'/std_masks/MNI152_T1_8mm_brain.nii.gz'));

for ss=1:length(subjects_to_do)
    spm_file_opt = [spmfilesdir preproc_name '_' session_name '_session' num2str(ss)];
    % Band pass filter: 
    S      = [];
    S.D    = spm_file_opt;
    S.band = 'bandpass';
    S.freq = freq;
    D = spm_eeg_filter(S);
    % move into bfdir
    Dnew = D.move(bfdir);
    bf_files = fullfile(Dnew);
    % Beamform:   
    S = struct;
    S.modalities        = {'MEGGRAD'}; 
    S.timespan          = [0 Inf];
    S.pca_order         = 250;
    S.type              = 'Scalar';
    S.inverse_method    = 'beamform';
    S.prefix            = '';
    D = spm_eeg_load(bf_files);
    osl_inverse_model(D,mni_coords,S);
end


%%%%%%%%%%%%%%%%%%%%%%%%%
%% parcellate and orthogonalise
% leakage correction can be done using:
%   - the method described in Colclough et al (2015) , referred to as 'symmetric'
%   - the method described in Pasqual-Marqui et al (2017), referred to as 'innovations_mar'
%       in which case we need to specify the option 'innovations_mar_order'
%   - just 'none'

orthogonalisation = 'innovations_mar';
innovations_mar_order = 14; 

parcdir = [codedir '/parcellations'];
parcfile = [parcdir '/fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz'];
parcname = '42ROIs_with_separatePCC';
parcprefix = [parcname '_' orthogonalisation num2str(innovations_mar_order) '_'];

bf_files = cell(length(subjects_to_do),1);
for ss=1:length(subjects_to_do)
    spm_file_opt = [spmfilesdir preproc_name '_' session_name '_session' num2str(ss)];    
    [~, fname] = fileparts(spm_file_opt);
    bf_files{ss} = [bfdir fname];
    bf_files{ss} = prefix(bf_files{ss},'f');
end

parcellated_Ds = cell(length(subjects_to_do),1);
for ss = 1:length(subjects_to_do)
    S                   = [];
    S.D                 = bf_files{ss};
    S.parcellation      = parcfile;
    S.orthogonalisation = orthogonalisation;
    S.innovations_mar_order = innovations_mar_order;
    S.method            = 'spatialBasis';
    S.normalise_voxeldata = 0;
    S.prefix = parcprefix;
    [parcellated_Ds{ss},parcelWeights,parcelAssignments] = osl_apply_parcellation(S);
    parcellated_Ds{ss}.parcellation.weights = parcelWeights;
    parcellated_Ds{ss}.parcellation.assignments = parcelAssignments; 
    parcellated_Ds{ss}.save;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare data to run the HMM
% Although the HMM-MAR toolbox can process SPM, we save each subject in a
% .mat file for simplicity, with a matrix X of dimension (time by regions).
% We also need to save a vector T with the length of each session. 

addpath(genpath(hmmdir))

mat_files = cell(length(subjects_to_do),1);
T_all = cell(length(subjects_to_do),1);
for ss = 1:length(subjects_to_do)
    mat_files{ss} = [matrixfilesdir 'subject' num2str(ss) '.mat'];
    [~,T_ss] = read_spm_file(parcellated_Ds{ss},mat_files{ss});
    T_all{ss} = T_ss;
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%
%% Resolve dipole sign ambiguity

options_signflip = [];
options_signflip.maxlag = 7; 
options_signflip.verbose = 0;

flips = findflip(mat_files,T_all,options_signflip);
flipdata(mat_files,T,flips);

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run the HMM

outputfile = [outputdir '/hmm_analysis.mat'];
embeddedlag = 7; K = 12; Hz = 250; ndim = 42;

options = struct();
options.order = 0;
options.zeromean = 1;
options.covtype = 'full';
options.embeddedlags = -embeddedlag:embeddedlag;
options.pca = ndim*2;
options.K = K;
options.Fs = Hz;
options.verbose = 1;
options.onpower = 0; 
options.standardise = 1;
options.standardise_pc = options.standardise;
options.inittype = 'HMM-MAR';
options.cyc = 100;
options.initcyc = 10;
options.initrep = 3;

% stochastic options
options.BIGNinitbatch = 5;
options.BIGNbatch = 5;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

[hmm, Gamma, ~, vpath] = hmmmar(mat_files,T_all,options);
save(outputfile,'hmm','Gamma','vpath')

% Investigate the amount of variance explained 
explained_var = explainedvar_PCA(mat_files,T_all,options);
figure; plot(explained_var,'LineWidth',3);


%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get the auto-covariance matrix, or cross-correlation matrix, for each state

k = 1; 
C = getAutoCovMat(hmm,k);
figure; imagesc(C); 

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute the spectra, at the group level and per subject

N = length(subjects_to_do);
options_mt = struct('Fs',Hz); % Sampling rate - for the 25subj it is 300
options_mt.fpass = [1 45];  % band of frequency you're interested in
options_mt.tapers = [4 7]; % taper specification - leave it with default values
options_mt.p = 0; %0.01; % interval of confidence  
options_mt.win = 2 * Hz; % multitaper window
options_mt.to_do = [1 0]; % turn off pdc
options_mt.order = 0;
options_mt.embeddedlags = -7:7;

% average
fitmt = hmmspectramt(mat_files,T_all,Gamma,options_mt);

% per subject
fitmt_subj = cell(N,1);
d = length(options_mt.embeddedlags) - 1; 
acc = 0; for n=1:N
    load(mat_files{n});
    gamma = Gamma(acc + (1:(sum(T_all{n})-length(T_all{n})*d)),:);
    acc = acc + size(gamma,1);
    fitmt_subj{n} = hmmspectramt(X,T_all{n},gamma,options_mt);
    fitmt_subj{n}.state = rmfield(fitmt_subj{n}.state,'ipsd');
    fitmt_subj{n}.state = rmfield(fitmt_subj{n}.state,'pcoh');
    fitmt_subj{n}.state = rmfield(fitmt_subj{n}.state,'phase');
    disp(['Subject ' num2str(n)])
end
save(outputfile,'fitmt_subj','fitmt','-append') 


%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do an automatic spectral factorisation to find spectrally-defined networks 
% (see paper for more info)
% Note: manual inspection of the spectral modes (i.e. what is contained in sp_profiles_wb
% and sp_profiles_4b) is **strongly** recommended. This is an algorithmic
% solution and there is no theoretical guarantee of getting a sensible
% result.

% Get the three bands depicted in the paper (the 4th is essentially capturing noise)
options_fact = struct();
options_fact.Ncomp = 4; 
options_fact.Base = 'coh';
[fitmt_group_fact_4b,sp_profiles_4b,fitmt_subj_fact_4b] = spectdecompose(fitmt_subj,options_fact);
save(outputfile,'fitmt_subj_fact_4b','fitmt_group_fact_4b','sp_profiles_4b','-append') 
% Get the wideband maps (the second is capturing noise)
options_fact.Ncomp = 2; 
[fitmt_group_fact_wb,sp_profiles_wb,fitmt_subj_fact_wb] = spectdecompose(fitmt_subj,options_fact);
save(outputfile,'fitmt_subj_fact_wb','fitmt_group_fact_wb','sp_profiles_wb','-append') 

% check if the spectral profiles make sense, if not you might like to repeat
figure; 
subplot(1,2,1); plot(sp_profiles_4b,'LineWidth',2)
subplot(1,2,2); plot(sp_profiles_wb,'LineWidth',2)

%% Do statistical testing on the spectral information

fitmt_subj_fact_1d = cell(N,1);
for n = 1:N
    fitmt_subj_fact_1d{n} = struct();
    fitmt_subj_fact_1d{n}.state = struct();
    for k = 1:K % we don't care about the second component
        fitmt_subj_fact_1d{n}.state(k).psd = fitmt_subj_fact_wb{n}.state(k).psd(1,:,:);
        fitmt_subj_fact_1d{n}.state(k).coh = fitmt_subj_fact_wb{n}.state(k).coh(1,:,:);
    end
end
tests_spectra = specttest(fitmt_subj_fact_1d,5000,1,1);
significant_spectra = spectsignificance(tests_spectra,0.01);

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get maps in both NIFTI and CIFTI formats

workbenchdir = [mydir '/workbench/bin_macosx64']; % set this to yours
path1 = getenv('PATH');
if ~contains(path1,workbenchdir)
    path1 = [path1 ':' workbenchdir];
    setenv('PATH', path1);
end

maskfile = [codedir '/std_masks/MNI152_T1_8mm_brain'];
parcdir = [codedir '/parcellations'];
parcfile = [parcdir '/fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz'];

spatialMap = parcellation(parcfile);
spatialMap = spatialMap.to_matrix(spatialMap.weight_mask);

% compensate the parcels to have comparable weights 
for j=1:size(spatialMap,2) % iterate through regions : make max value to be 1
    spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
end

% Wideband
mapfile = [mapsdir '/state_maps_wideband'];
map = zeros(size(spatialMap,1),length(fitmt_group_fact_wb.state));
for k = 1:12
    psd = diag(squeeze(fitmt_group_fact_wb.state(k).psd(1,:,:)));
    map(:,k) = spatialMap * psd;
end
[mask,res,xform] = nii.load(maskfile);
nii.save(matrix2vols(map,mask),res,xform,mapfile);
osl_render4D([mapfile '.nii.gz'],'savedir','weighted_maps/',...
    'interptype','trilinear','visualise',false)
% centered by voxel across states
mapfile = [mapsdir '/state_maps_wideband_centered'];
map = map - repmat(mean(map,2),1,size(map,2));
nii.save(matrix2vols(map,mask),res,xform,mapfile);
osl_render4D([mapfile '.nii.gz'],'savedir','weighted_maps/',...
    'interptype','trilinear','visualise',false)

% Per frequency band
for fr = 1:3
    mapfile = [mapsdir '/state_maps_band' num2str(fr)];
    map = zeros(size(spatialMap,1),length(fitmt_group_fact_4b.state));
    for k = 1:12
        psd = diag(squeeze(fitmt_group_fact_4b.state(k).psd(fr,:,:)));
        map(:,k) = spatialMap * psd;
    end
    nii.save(matrix2vols(map,mask),res,xform,mapfile);
    osl_render4D([mapfile '.nii.gz'],'savedir','weighted_maps/',...
        'interptype','trilinear','visualise',false)
    % centered by voxel across states
    mapfile = [mapsdir '/state_maps_band' num2str(fr) '_centered'];
    map = map - repmat(mean(map,2),1,size(map,2));
    nii.save(matrix2vols(map,mask),res,xform,mapfile);
    osl_render4D([mapfile '.nii.gz'],'savedir','weighted_maps/',...
        'interptype','trilinear','visualise',false)
end


%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute state life times, state interval times, 
% fractional occupancies and switching rate

LifeTimes = getStateLifeTimes (vpath,T_all,options,5);
Intervals = getStateIntervalTimes (vpath,T_all,options,5);
FO = getFractionalOccupancy (Gamma,T_all,options);
switchingRate =  getSwitchingRate(Gamma,T,options);

save(outputfile,'LifeTimes','Intervals','FO','switchingRate','-append')

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get glass connectivity brains
% doing data-driven thresholding
osldir = 1;
addpath([codedir '/ohba-external/netlab3.3/netlab'])
addpath([codedir '/ohba-external/osl-core/util'])
parcdir = [codedir '/parcellations'];
parcfile = [parcdir '/fmri_d100_parcellation_with_3PCC_ips_reduced_2mm_ss5mm_ds8mm_adj.nii.gz'];

K = length(fitmt_group_fact_4b.state); ndim = 42; 
spatialRes = 8; edgeLims = [4 8];

% wideband
M = zeros(42); 
for k = 1:K
    M = M + squeeze(abs(fitmt_group_fact_wb.state(k).coh(1,:,:))) / K;
end
for k = 1:K
    graph = squeeze(abs(fitmt_group_fact_wb.state(k).coh(1,:,:)));
    graph = (graph - M);  
    tmp = squash(triu(graph));
    inds2 = find(tmp>1e-10);
    data = tmp(inds2);
    S2 = [];
    S2.data = data;
    S2.do_fischer_xform = false;
    S2.do_plots = 1;
    S2.pvalue_th = 0.01/length(S2.data);
    graph_ggm = teh_graph_gmm_fit(S2); 
    th = graph_ggm.normalised_th;
    graph = graph_ggm.data';
    graph(graph<th) = NaN;
    graphmat = nan(ndim, ndim);
    graphmat(inds2) = graph;
    graph = graphmat;
    p = parcellation(parcfile);
    spatialMap = p.to_matrix(p.weight_mask);
    % compensate the parcels to have comparable weights
    for j=1:size(spatialMap,2) % iterate through regions : make max value to be 1
        spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
    end
    p.weight_mask = p.to_vol(spatialMap);
    [h_patch,h_scatter] = p.plot_network(graph,th);
end
    
% Per frequency band
for fr = 1:3
    M = zeros(42);
    for k = 1:K
        M = M + squeeze(abs(fitmt_group_fact_4b.state(k).coh(fr,:,:))) / K;
    end
    for k = 1:K
        graph = squeeze(abs(fitmt_group_fact_4b.state(k).coh(fr,:,:)));
        graph = (graph - M);
        tmp = squash(triu(graph));
        inds2 = find(tmp>1e-10);
        data = tmp(inds2);
        S2 = [];
        S2.data = data;
        S2.do_fischer_xform = false;
        S2.do_plots = 1;
        S2.pvalue_th = 0.01/length(S2.data);
        graph_ggm = teh_graph_gmm_fit(S2);
        th = graph_ggm.normalised_th;
        graph = graph_ggm.data';
        graph(graph<th) = NaN;
        graphmat = nan(ndim, ndim);
        graphmat(inds2) = graph;
        graph = graphmat;
        p = parcellation(parcfile);
        spatialMap = p.to_matrix(p.weight_mask);
        % compensate the parcels to have comparable weights
        for j=1:size(spatialMap,2) % iterate through regions : make max value to be 1
            spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
        end
        p.weight_mask = p.to_vol(spatialMap);
        [h_patch,h_scatter] = p.plot_network(graph,th);
    end
end

rmdir([codedir '/ohba-external/netlab3.3/netlab'])
rmpath([codedir '/ohba-external/osl-core/util'])



