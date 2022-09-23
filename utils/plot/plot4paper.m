function plot4paper(xtext,ytext)

a=axis;
if(nargin<2)ytext='';end;
set(gca,'fontsize',16);
set(gca,'LineWidth',2);
xlabel(xtext,'fontsize',16);
ylabel(ytext,'fontsize',16);
axis(a);