function plot_hmmspectra (spectra,err,coh,figure_number,colmap,channels)
% Plot PSD and Coherence or PDC for all states
%
% INPUT
% spectra: estimation from hmmspectramt or hmmspectramar
% err: whether or not to show confidence intervals (only if they were computed)
% coh: show coherence or PDC (latter only if it were computed)
% figure_number: number of figure to display
% colmap: (no. states by 3) matrix of colours
% channels: subset of the channels to show (default: all)
%
% Diego Vidaurre, University of Oxford (2019)
% Based on previous work from Alvaro Tejero-Cantero

if nargin < 2 || isempty(err), err = 0; end
if nargin < 3 || isempty(coh), coh = 1; end
if nargin < 4  || isempty(figure_number)
    figure;
else
    figure(figure_number);
end

K = length(spectra.state);
d = size(spectra.state(1).psd,2);

col = zeros(K,3);
if nargin < 5 || isempty(colmap)
    cm = colormap;
    r = round(linspace(1,64,K));
    for k = 1:K
        col(k,:) = cm(r(k),:);
    end
else
    if isstr(colmap) % an actual colormap
        cm = colormap(colmap);
        r = round(linspace(1,64,K));
        for k = 1:K
            col(k,:) = cm(r(k),:);
        end
    else % a matrix of colours
        col = colmap;
    end
end

if nargin < 6  || isempty(channels)
    channels = 1:d;
end

if length(channels)>10, warning('This is probably too many channels'); end


ci_color = [0.8 0.8 0.8]; % some grey.

for i = channels
    for j = channels % one panel per source-target ch combination
        
        ii = find(i==channels); jj = find(j==channels);
        subplot(length(channels),length(channels),(ii-1)*length(channels) + jj)
        
        if ii == jj   % on diagonal, show PSDs
            
            try
                if err
                    ciplot(s.psderr(1,:,i,i), s.psderr(2,:,i,i), s.f, ci_color);
                    hold on;
                end
            end
            
            hold on;
            m = 0;
            for k = 1:K
                s = spectra.state(k);
                if any(isnan(s.psd(:,i,i))), continue; end
                m = max(m,max(s.psd(:,i,i)));
                plot(s.f, (s.psd(:,i,i)), ...
                    'Color', col(k,:), 'LineWidth', 2);
            end
            
            yrng = [0 m];
            string_panel = ['(PSD ' num2str(i) ')'];
            text(0.7*s.f(end), 0.95*(yrng(2)-yrng(1))+yrng(1),string_panel)
            ylim([0 1.05*m]); xlim([s.f(1) s.f(end)])
            
            
            hold off
            
        else        % off diagonal, show PDCs or coherences
            if coh
                try
                    if err
                        ciplot(s.coherr(1,:,i,j), s.coherr(2,:,i,j), s.f, ci_color);
                        hold on;
                    end
                end
            else
                try
                    if err
                        ciplot(s.pdcerr(1,:,i,j), s.pdcerr(2,:,i,j), s.f, ci_color);
                        hold on;
                    end
                end
            end
            
            hold on
            m = 0;
            if coh
                for k = 1:K
                    s = spectra.state(k);
                    if any(isnan(s.coh(:,i,j))), continue; end
                    plot(s.f, (s.coh(:,i,j)),'Color', col(k,:), 'LineWidth', 2);
                    m = max(m,max(s.coh(:,i,j)));
                    string_panel = ['(Coh ' num2str(i) ',' num2str(j) ')'];
                end
            else
                for k = 1:K
                    s = spectra.state(k);
                    if any(isnan(s.pdc(:,i,j))), continue; end
                    plot(s.f, (s.pdc(:,i,j)),'Color', col(k,:), 'LineWidth', 2);
                    m = max(m,max(s.pdc(:,i,j)));
                    string_panel = ['(PDC ' num2str(i) '->' num2str(j) ')'];
                end
            end
            hold off
            
            %ylim([0,1]);
            
            yrng = [0 m];
            
            text(0.7*s.f(end), 0.95*(yrng(2)-yrng(1))+yrng(1),string_panel)
            ylim([0 1.05*m]); xlim([s.f(1) s.f(end)])
            
        end
        
        %if i == 1 && j == 1, ylabel('Power'); end
        
        if ii == 1, title(['Channel ' num2str(j)],'FontSize',10); end
        if ii == length(channels), xlabel('Frequency (Hz)'); end
        if jj == 1, ylabel(['Channel ' num2str(i)],'FontSize',10,'FontWeight','bold'); end
    end
    
end
end

