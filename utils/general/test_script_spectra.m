order = 3;
X = randn(2000,2); T = [250 250 250 250 250 250 250 250];
N = length(T);
options = struct();
options.K = 2;
options.order = order;
options.filter = [1 Inf];
options.Fs = 100;
options.inittype = 'random';
options.cyc =1; 
[hmm,Gamma] = hmmmar(X,T,options);

%%

for p = [0 0.99]
    for format = [0 1 2]
        
        options.p = p;
        options.to_do = [1 1];
        
        if format==0
            data = X; Tdata = T;
        elseif format==1
            data = cell(N,1); Tdata = cell(N,1);
            for j = 1:N, data{j} = X((j-1)*250+(1:250),:); Tdata{j} = 250; end
        else
            data = cell(N,1); Tdata = cell(N,1);
            for j = 1:N
                f = ['/tmp/blah_' num2str(j) '.mat'];
                data{j} = f;
                Y =  X((j-1)*250+(1:250),:);
                save(f,'Y');
                Tdata{j} = 250;
            end
        end
            
        fitmt = hmmspectramt(data,Tdata,Gamma,options);
        fitmar1 = hmmspectramar([],[],hmm,[],options);
        fitmar2 = hmmspectramar(data,Tdata,[],Gamma,options);
        
    end
end