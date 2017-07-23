function [] = plotfeterms(feterms,cycrange)
if nargin<2, cycrange = 1:size(feterms.loglik,2); end
f1 = -sum(feterms.loglik,1); % 1 x cyc
f2 = squeeze(sum(feterms.subjfe,1)); % 3 x cyc
f3 = feterms.statekl;
F = [f1(1,cycrange); f2(:,cycrange); f3(1,cycrange)];
subplot(2,1,1)
plot(cycrange,sum(F,1),'k','LineWidth',4);
subplot(2,1,2)
f1 = f1 - mean(f1);
for j = 1:3, f2 = f2 - mean(f2(j,:)); end
f3 = f3 - mean(f3);
F = [f1(1,cycrange); f2(:,cycrange); f3(1,cycrange)];
plot(cycrange,F,'LineWidth',2)
legend('Loglik','\Gamma entropy','\Gamma loglik','KL P/Pi','State KL')
set(gca,'ytick',[])
end