% Plotting figures

yregret1=importdata('/Users/sougatachaudhuri/Box Sync/Programs/OnlineRankingTop-1(NC)/DCG100.txt','\n');
yregret2=importdata('/Users/sougatachaudhuri/Box Sync/Programs/OnlineRankingTop-1(NC)/DCG250.txt','\n');
yregret3=importdata('/Users/sougatachaudhuri/Box Sync/Programs/OnlineRankingTop-1(NC)/DCG400.txt','\n');

% Print for only partial information graphs

xrange=10:1:length(yregret1);
yregret1=yregret1(10:end);
yregret2=yregret2(10:end);
yregret3=yregret3(10:end);


hand=plot(xrange,yregret1, 'r', xrange, yregret2, 'b', xrange, yregret3, 'g');
set(gca, 'fontsize',20);
set(hand,'LineWidth',2);
xlabel('Iterations');
ylabel('AverageRegret');
hleg = legend('Regret:K=100', 'Regret:K=200', 'Regret:K=400');
rect = [0.64, 0.13, .25, .40];
set(hleg, 'FontSize',20);

% Print for full information graphs

% xrange=1000:10: length(yregret1);
% yregret1=yregret1(xrange);
% yregret2=yregret2(xrange);
% 
% temp=yregret1(1);
% yregret1(:)=yregret1(:)*yregret2(1)/temp;
% %yrange1=regretfull(xrange);
% hand=plot(xrange,yregret1,'-*',xrange,yregret2,'-o');
% set(gca, 'fontsize',20);
% set(hand,'LineWidth',2);
% xlabel('Iterations');
% ylabel('AverageRegret');
% hleg = legend('Regret-FullFeedback','Regret:Top-1Feedback');
% set(hleg, 'FontSize',20);
% %plot(xrange, regretfull(1000:T),'blue');

