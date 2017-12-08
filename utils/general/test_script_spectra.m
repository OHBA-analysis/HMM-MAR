hmm = cell(2,1); Gamma = cell(2,1); 
options = cell(2,1); options_s = cell(2,1); 


X = randn(2000,2); T = [250 250 250 250 250 250 250 250];
N = length(T);
options{1} = struct();
options{1}.K = 2;
options{1}.order = 3;
options{1}.filter = [1 Inf];
options{1}.Fs = 100;
options{1}.inittype = 'random';
options{1}.cyc =1; 
[hmm{1},Gamma{1}] = hmmmar(X,T,options{1});

options{2} = struct();
options{2}.K = 2;
options{2}.order = 0;
options{2}.embeddedlags = -3:3;
options{2}.filter = [1 Inf];
options{2}.Fs = 100;
options{2}.inittype = 'random';
options{2}.cyc =1;
[hmm{2},Gamma{2}] = hmmmar(X,T,options{2});

options_s{1} = struct();
options_s{1}.Fs = 100;
options_s{1}.filter = [1 Inf];
options_s{1}.order = 3;
options_s{1}.K = 2;


options_s{2} = struct();
options_s{2}.Fs = 100;
options_s{2}.filter = [1 Inf];
options_s{2}.order = 0;
options_s{2}.embeddedlags = -3:3;
options_s{2}.K = 2;


%%
for opt = 1:2
    for p = 0 %[0 0.99]
        for format = [0 1 2]
            
            options_s{opt}.p = p;
            options_s{opt}.to_do = [1 1];
            
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
            
            fitmt = hmmspectramt(data,Tdata,Gamma{opt},options_s{opt});
            fitmar1 = hmmspectramar([],[],hmm{opt},[],options_s{opt});
            if opt==1 % not valid for embedded
                fitmar2 = hmmspectramar(data,Tdata,[],Gamma{opt},options_s{opt});
            end
        end
    end
end