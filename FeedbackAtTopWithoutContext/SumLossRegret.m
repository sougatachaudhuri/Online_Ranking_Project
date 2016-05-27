% Program to apply regret bound algorithm on SumLoss (Ranking with
% Diversity paper)

function SumLossRegret()
T=9964;m=10;p=0.5;
%adversaryrel=binornd(1,p,m,T);
%Reading training dataset (Fold1, Ohsumed)
X=read_traindataset();
query=unique(X(:,1));
mi=find(X(:,1)==query(2),1,'first');
ma=find(X(:,1)==query(2),1,'last');
%Extracting 10 documents from 1st query
data=X(mi:ma,2:(size(X,2)));
rows=[1,3,4,5,6,7,8, 9,10,11];
docs=data(rows,1:size(data,2));
actualrelevance= docs(:,1);
actualrelevance=sort(actualrelevance,'descend');
%Creating 10000 corrupted copies of the relevance vector
for i=1:10000
    rel(1:5)= floor(2*(actualrelevance(1:5)- unifrnd(0,.7,5,1)));
    rel(6:10)= floor(2*(actualrelevance(6:10)+ unifrnd(0,0.7,5,1)));
    adversaryrel(:,i)= rel ;
end

%Creating constants for RT-1F algorithm
K=212;blocksize=47;eta=46;totalruns=15; % T= K*blocksize, eta=1/epsilon
% m=10;T=1;K=1;blocksize=1;eta=46;totalruns=1;
num=1:1:m;num=transpose(num);SumLossFinal=repmat(0,T,1);
% An outer loop will simulate the expectation over the relevance vectors.
% Inner loop makes one pass over T relevance vectors.
for runs=1:totalruns
    cumloss=0;count=0;estimatedrel=repmat(0,10,1);blockrel=repmat(0,10,1);
    for k=1:K
        estimatedrel=estimatedrel+blockrel;
        %Sample 10 uniform time points in the block B_k
        tempdata=((k-1)*blocksize+1):1:(k*blocksize); 
        explorepoints=randsample(tempdata,m);
        for t= ((k-1)* blocksize +1): (k*blocksize)
            count=count+1; 
            idx=find(explorepoints(:)==t);
            if(isempty(idx)==0)
               % Find permutation which places object idx at top and then calculate the loss
               objects=randperm(10);
               idx1=find(objects(:)==1);
               temp=objects(idx);
               objects(idx)=objects(idx1);
               objects(idx1)=temp;
               % Receive actual relevance of top placed object
               blockrel(idx)= adversaryrel(idx,count);
               % Calculate loss in the round
               cumloss=cumloss+objects*adversaryrel(:,count);
               loss(t)=cumloss;
            else
               % Output permutation according to Kalai-Vempala Algorithm
               p=unifrnd(0,eta,m,1);
               temprel=estimatedrel+p;
               temprel1=[temprel num];
               temprel1= flipdim(sortrows(temprel1,1),1);
               objects(temprel1(:,2))=num;
               % Calculate loss in the round
               cumloss=cumloss+objects*adversaryrel(:,count);
               loss(t)=cumloss;
            end
        end  
       % Dont need to put blockrel in a vector as it is already formed in
       % vector form
    end
    SumLossFinal=SumLossFinal + transpose(loss);
end
SumLossFinal=SumLossFinal/totalruns;

%Loss under full information
totalrel=repmat(0,10,1);
for t=1:T
    p=unifrnd(0,eta,m,1);
    totalreltemp=totalrel+p;
    totalrel1=[totalreltemp num];
    totalrel1= flipdim(sortrows(totalrel1,1),1);
    objects(totalrel1(:,2))=num;
    cumloss=cumloss+objects*adversaryrel(:,t);
    SumLossFullInfo(t)=cumloss;
    totalrel=totalrel + adversaryrel(:,t);
end
SumLossFullInfo=transpose(SumLossFullInfo);
%Calculating best permutation in hindsight
cumadversaryrel=sum(adversaryrel(:,1:T),2);
temprel=[cumadversaryrel num];
temprel= flipdim(sortrows(temprel,1),1);
objects(temprel(:,2))=num;cumloss=0;
for t=1:T
    cumloss=cumloss+objects*adversaryrel(:,t);
    MinLossFinal(t)=cumloss;
end
MinLossFinal=transpose(MinLossFinal);
%Calculating Regret
time=1:1:T;
time=transpose(time);
global regretfull regretpartial;
regretpartial=(SumLossFinal-MinLossFinal)./time;
regretfull=(SumLossFullInfo-MinLossFinal)./time;
xrange=1000:1:T;
yrange=regretpartial(1000:T);
hand=plot(xrange,yrange, 'r');
set(gca, 'fontsize',20);
set(hand,'LineWidth',2);
xlabel('Iterations');
ylabel('AverageRegret');
hleg = legend('AvgRegret@Top-1');
set(hleg, 'FontSize',20);
figure();
xrange=1000:90:T;
yrange=regretpartial(xrange);
%yrange=regretpartial(1000:T);
temp=regretfull(1000);
regretfull(1000:T)=regretfull(1000:T)*regretpartial(1000)/temp;
yrange1=regretfull(xrange);
hand=plot(xrange,yrange,'-*',xrange,yrange1,'-o');
set(gca, 'fontsize',20);
set(hand,'LineWidth',2);
xlabel('Iterations');
ylabel('AverageRegret');
hleg = legend('AvgRegret@Top-1','AvgRegret@FullInfo');
set(hleg, 'FontSize',20);
%plot(xrange, regretfull(1000:T),'blue');
end

function [X]=read_traindataset()
f = fopen('/Users/sougatachaudhuri/Box Sync/Programs/OHSUMED/QueryLevelNorm/Fold1/train.txt');
  X = zeros(2e5,0);
  R = zeros(1e5,1);
  Q = zeros(1e5,1);
  qid = '';
  i = 0; q = 0;
  while 1
    l = fgetl(f);
    if ~ischar(l), break; end;
    i = i+1; 
    R(i,1)= sscanf(l,'%d',1);
    
    [lab,  foo1, foo2, ind] = sscanf(l,'%d qid:',1); l(1:ind-1)=[];
    [nqid, foo1, foo2, ind] = sscanf(l,'%s',1); l(1:ind-1)=[];Q(i,1)=str2num(nqid); 
    if ~strcmp(nqid,qid)
      q = q+1;
      qid = nqid;
      Y{q} = lab;
    else 
      Y{q} = [Y{q}; lab];
    end;
    tmp = sscanf(l,'%d:%f'); 
    X(i,tmp(1:2:end)) = tmp(2:2:end);
  end;
  X = X(1:i,:); R=R(1:size(X,1),1);Q=Q(1:size(X,1),1);
  X=cat(2,Q,R,X);
  fclose(f);
end

% Extra Code
% normfactors= sqrt(sum(docs.^2, 2));
% normfactormatrix=diag(1./normfactors);
% normdocs= normfactormatrix*docs;
% %Generating function parameter
% w=normrnd(0,.5,size(docs,2),1);
% w=w./norm(w);
% scores=normdocs*w;
% disp(scores);