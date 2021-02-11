function ord = plot_Gamma (Gamma,T,continuous,order_trials,behaviour,cm)

if nargin < 2 || isempty(T), T = size(Gamma,1); end
if nargin < 3, continuous = length(T)==1 || any(T(1)~=T); end % show it as continuous data? 
if nargin < 4, order_trials = false; end % order the trials?
if nargin < 5, behaviour = []; end % behaviour with respect to which order the trials 
if nargin < 6, cm = colormap; end % colormap

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    T = cell2mat(T); T = T(:);
end

N = length(T); K = size(Gamma,2); 
d = (sum(T) - size(Gamma,1)) / N;
T = T - d; 

if continuous
    
    ord = [];
    t1 = min(size(Gamma,1),1000);
    area(Gamma(1:t1,:)); ylim([0 1]); xlim([0 t1])
    
else
    
    if any(T(1)~=T)
        ind = [];
        for n = 1:length(T)
            ind = [ind ((1:min(T))+sum(T(1:n-1))) ];
        end
        Gamma = Gamma(ind,:);
        T(:) = min(T);
    end

    colors = cm(round(linspace(1,size(cm,1),K)),:);
    
    keep = true(N,1);
    % Set the ordering
    ord = 1:N;
    if length(order_trials) > 1
        ord = order_trials;
    elseif order_trials && ~isempty(behaviour)
        keep = ~isnan(behaviour); % for which trials do we have behaviour?
        [~,ord] = sort(behaviour(keep),'ascend'); % order by behaviour
    elseif order_trials
        A = pca(reshape(permute(reshape(Gamma,[T(1) N K]),[2 1 3]),[N T(1)*K])',...
            'NumComponents',1);
        [~,ord] = sort(A);
    end
        
    GammaCol = zeros(size(Gamma,1),3);
    for k = 1:K
        these = sum(repmat(Gamma(:,k),1,K-1) > Gamma(:,setdiff(1:K,k)),2) == K-1;
        GammaCol(these,:) = repmat(colors(k,:),sum(these),1);
    end
    GammaCol = reshape(GammaCol,[T(1) length(T) 3]); % -> (time by trials by states)
    GammaCol = GammaCol(:,keep,:); % only trials with behaviour
    GammaCol = permute(GammaCol(:,ord,:),[2 1 3]); % -> (trials by time by states)
    subplot(4,1,1:3)
    imagesc(1:T(1),1:size(GammaCol,1),GammaCol)
    xlabel('Time (s)'); ylabel('Trials')
    set(gca,'FontSize',16)
    subplot(4,1,4)
    Gamma = reshape(Gamma,[T(1) length(T) K]); 
    Gamma = squeeze(mean(Gamma(:,keep,:),2));
    hold on
    for k = 1:K
        plot(Gamma(:,k),'Color',colors(k,:),'LineWidth',3)
    end
    hold off
    xlabel('Time (s)'); xlim([1 T(1)]);
    set(gca,'FontSize',16)
end
    
end