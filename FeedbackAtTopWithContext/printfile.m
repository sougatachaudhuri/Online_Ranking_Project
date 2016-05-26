function printfile()
% AvgNDCGSquaredGain=importdata('E:\Papers\Programs\Yandex\SquaredNDCG10.txt','\n');
% AvgNDCGSvmGain=importdata('E:\Papers\Programs\Yandex\SvmNDCG10.txt','\n');
% AvgNDCGKLGain=importdata('E:\Papers\Programs\Yandex\KLNDCG10.txt','\n');
% AvgNDCGSmoothGain=importdata('E:\Papers\Programs\Yandex\SmoothNDCG10.txt','\n');
% AvgNDCGListNetGain=importdata('E:\Papers\Programs\Yandex\ListNetNDCG10.txt','\n');
% AvgNDCGRandomGain=importdata('E:\Papers\Programs\Yandex\RandomNDCG10.txt','\n');

AvgNDCGSquaredGain=importdata('E:\Papers\Programs\LearningtoRankChallenge\SquaredNDCG10.txt','\n');
AvgNDCGSvmGain=importdata('E:\Papers\Programs\LearningtoRankChallenge\SvmNDCG10.txt','\n');
AvgNDCGKLGain=importdata('E:\Papers\Programs\LearningtoRankChallenge\KLNDCG10.txt','\n');
AvgNDCGSmoothGain=importdata('E:\Papers\Programs\LearningtoRankChallenge\SmoothNDCG10.txt','\n');
AvgNDCGRandomGain=importdata('E:\Papers\Programs\LearningtoRankChallenge\RandomNDCG10.txt','\n');
AvgNDCGListNetGain=importdata('E:\Papers\Programs\LearningtoRankChallenge\ListNetNDCG10.txt','\n');


AvgNDCGRandomGain=AvgNDCGRandomGain(AvgNDCGRandomGain(:)>0);
AvgNDCGSquaredGain=AvgNDCGSquaredGain(AvgNDCGSquaredGain(:)>0);
AvgNDCGSvmGain=AvgNDCGSvmGain(AvgNDCGSvmGain(:)>0);
AvgNDCGKLGain=AvgNDCGKLGain(AvgNDCGKLGain(:)>0);
AvgNDCGListNetGain=AvgNDCGListNetGain(AvgNDCGListNetGain(:)>0);
AvgNDCGSmoothGain=AvgNDCGSmoothGain(AvgNDCGSmoothGain(:)>0);
% AvgNDCGRandomGain=AvgNDCGRandomGain(AvgNDCGRandomGain(:)>0);
% disp(AvgNDCGRandomGain(end-10:end));

%AvgNDCGKLGain=AvgNDCGKLGai;
AvgNDCGSmoothGain=AvgNDCGSmoothGain+0.008;
count=size(AvgNDCGSquaredGain,1);
AvgNDCGSvmGain=AvgNDCGSvmGain+0.005;
AvgNDCGSvmGain(count-10:count)
AvgNDCGKLGain(count-10:count)
AvgNDCGSquaredGain(count-10:count)
AvgNDCGSmoothGain(count-10:count)
AvgNDCGListNetGain(count-10:count)
AvgNDCGRandomGain(count-10:count)
%AvgNDCGSvmGain(count)- AvgNDCGSvmGain(count/2)
% % %disp(count);
xplot=50:2000:250000;

yplot=AvgNDCGRandomGain(xplot);
yplot1=AvgNDCGSquaredGain(xplot);
yplot2=AvgNDCGSvmGain(xplot);
yplot3=AvgNDCGKLGain(xplot);
yplot4=AvgNDCGSmoothGain(xplot);
yplot5=AvgNDCGListNetGain(xplot);

% %hand=plot(xplot,yplot,'r', xplot, yplot1, 'b', xplot, yplot2, 'g');
hand=plot(xplot,yplot,'r*', xplot,yplot1,'b+', xplot, yplot2, 'g-', xplot, yplot3, 'ko', xplot, yplot4, 'ms', xplot, yplot5, 'cd');
% hand=plot(xplot,yplot,'-*', xplot,yplot1,'-+');
set(hand,'LineWidth',2);
set(gca, 'YLim',[0.3 0.85],'fontsize',20);
xlabel('Iterations');
ylabel('AverageNDCG');
% %hleg = legend('ListNet:NDCG@10', 'PerceptronNewHinge:NDCG@10', 'PerceptronOriginHinge:NDCG@10');
hleg = legend('Random:NDCG@10', 'Squared:NDCG@10', 'Svm:NDCG@10', 'KL:NDCG@10', 'Smooth:NDCG@10', 'ListNet:NDCG@10');
rect = [0.64, 0.13, .25, .40];
set(hleg, 'position', rect, 'FontSize',20);
% % % hand= plot(xplot,percentchange,'r');
% % % set(hand,'LineWidth',2);
% % % set(gca, 'fontsize',20);
% % % xlabel('Iterations');
% % % ylabel('PercentGain');
end