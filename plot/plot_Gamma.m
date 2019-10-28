function ord = plot_Gamma (Gamma,T,take_mean,order_trials,behaviour,cm)

if nargin < 3, take_mean = false; end 
if nargin < 4, order_trials = false; end 
if nargin < 5, behaviour = []; end 
if nargin < 6, cm = colormap; end 

N = length(T); K = size(Gamma,2); 
d = (sum(T) - size(Gamma,1)) / N;
T = T - d; 

if take_mean
    
    area(Gamma); ylim([0 1]); xlim([0 size(Gamma,1)])
    
else
    
    if any(T(1)~=T)
        error('All trials need to have the same number of time points when order_trials==1'); 
    end

    colors = cm(round(linspace(1,64,K)),:);
    
    keep = true(N,1);
    if ~isempty(behaviour)
        keep = ~isnan(behaviour); % which trials the subject pressed the button?
        if order_trials
            [~,ord] = sort(behaviour(keep),'ascend'); % order by behaviour
        else
            ord = 1:sum(keep);
        end
    elseif order_trials
            A = pca(reshape(permute(reshape(Gamma,[T(1) N K]),[2 1 3]),[N T(1)*K])',...
                'NumComponents',1);
            [~,ord] = sort(A); 
    else
        ord = 1:N;
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