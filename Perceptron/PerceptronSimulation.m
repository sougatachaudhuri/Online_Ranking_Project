% This code is to generate the 1st graph in perceptron paper.
% It compares performance of the two perceptron algorithms and  online ListNet
% on margin separable simulated dataset.

function PerceptronSimulation()
tic
% Declaration of parameters
u=ones(10,1);wrankSLAM=zeros(10,1); wrankOA=zeros(10,1); wrankListNet=zeros(10,1);

% The different relevance levels.
rel=[4;4;4;3;3;3;2;2;2;2;1;1;1;1;1;0;0;0;0;0];

%rel=[2;1;1;1;1;0;0;0;0;0];
%Declaration of further parameters
T=1000;delta=1; 
CumNDCGSLAMGain=zeros(T,1); AvgNDCGSLAMGain=zeros(T,1);
CumNDCGOAGain=zeros(T,1); AvgNDCGOAGain=zeros(T,1);
CumNDCGListNetGain=zeros(T,1); AvgNDCGListNetGain=zeros(T,1);

% Iterations over time horizon T
for t=1:T
    % Margin separable data generation
    data= DataGeneration(u,rel);
    
    %Perceptron based on SLAM family
    loss= NDCGLoss(wrankSLAM,data);
   
    if(t==1)
        CumNDCGSLAMGain(1)=1-loss;
        AvgNDCGSLAMGain(1)=CumNDCGSLAMGain(1);
        
    else
        CumNDCGSLAMGain(t)= CumNDCGSLAMGain(t-1) + 1- loss;
        AvgNDCGSLAMGain(t)=CumNDCGSLAMGain(t)/t;
    end
    %Calculating Gradient
    if(loss>0)
       gradSLAM=transpose(calculateSLAMgradient(wrankSLAM,delta,data));
       wrankSLAM=wrankSLAM - 1/sqrt(t)*gradSLAM;
    end
    
    %Purely Online Algorithm
    loss= NDCGLoss(wrankOA,data);
    
    if(t==1)
        CumNDCGOAGain(1)=1-loss;
        AvgNDCGOAGain(1)=CumNDCGOAGain(1);
        
    else
        CumNDCGOAGain(t)= CumNDCGOAGain(t-1) + 1- loss;
        AvgNDCGOAGain(t)=CumNDCGOAGain(t)/t;
    end
    %Calculating Gradient
    if(loss>0)
       gradOA=transpose(calculateOAgradient(wrankOA,data,loss));
       wrankOA=wrankOA - 1/sqrt(t)*gradOA;
    end
    
    
    %Online version of ListNet
    loss=NDCGLoss(wrankListNet,data);
        if (t==1)
            CumNDCGListNetGain(1)=1-loss;
            AvgNDCGListNetGain(1)=CumNDCGListNetGain(1);   
        else
            CumNDCGListNetGain(t)=CumNDCGListNetGain(t-1)+ 1-loss;
            AvgNDCGListNetGain(t)=CumNDCGListNetGain(t)/t;     
        end
            wrankListNet=wrankListNet -1/sqrt(t)* calculateListNetgradient(wrankListNet,data);
end

%Plotting routine
xplot=1:1:400;
yplot=AvgNDCGListNetGain(xplot);
yplot1=AvgNDCGSLAMGain(xplot);
yplot2=AvgNDCGOAGain(xplot);
disp(AvgNDCGOAGain(1:20));
disp(AvgNDCGSLAMGain(1:20));
disp(AvgNDCGListNetGain(1:20))
hand=plot(xplot,yplot,'-+', xplot,yplot1,'-*', xplot, yplot2, '--');
set(hand,'LineWidth',2);
set(gca, 'fontsize',20);
xlabel('Iterations');
ylabel('AverageNDCG');
hleg = legend('ListNet:NDCG@10', 'PerceptronSLAM:NDCG@10', 'PureOnline:NDCG@10');
rect = [0.65, 0.12, .15, .25];
set(hleg, 'position', rect, 'FontSize',30);
end

% Function to generate margin separable data matrix
function [X]= DataGeneration(u,rel)
X=zeros(20,10);
for i=1:3
    X(i,:)= normrnd(7,1,1,10);  
end
score=min(X(1:3,:)*u);

for i=4:6
    flag=0;
    while(flag==0)
        x= normrnd(5,1,1,10);
        if ((score- (x*u))>1)
            X(i,:)=x;
            flag=1;
        end
    end
end
score=min(X(4:6,:)*u);

for i=7:10
    flag=0;
    while(flag==0)  
        x= normrnd(3,1,1,10);
        if ((score- (x*u))>1)
            X(i,:)=x;
            flag=1;
        end
    end
end
score=min(X(7:10,:)*u);

for i=11:15
    flag=0;
    while(flag==0)  
        x= normrnd(1,1,1,10);
        if ((score- (x*u))>1)
            X(i,:)=x;
            flag=1;
        end
    end
end
score=min(X(11:15,:)*u);

for i=16:20
    flag=0;
    while(flag==0)  
        x= normrnd(-1,1,1,10);
        if ((score- (x*u))>1)
            X(i,:)=x;
            flag=1;
        end
    end
end

X=cat(2,rel,X);
end


% Gradient Function for ListNet Loss
function [grad]=calculateListNetgradient(wrank,data)
% Calculating the gradient from Eq.6 of ListNet paper.
data=transpose(data); len=size(data,1); 

% The first part of the equation
denom=sum(exp(data(1,:)));
num=-1* data(2:len,:)*transpose(exp(data(1,:)));
ratio1= num/denom;

% The 2nd part of the equation
denom=sum(exp(transpose(wrank)* data(2:len,:)));
num= data(2:len,:)*transpose(exp(transpose(wrank)* data(2:len,:)));
ratio2=num/denom;

grad= ratio1+ ratio2;
end

%Gradient Function for SLAM Loss
function [grad]= calculateSLAMgradient(wrank,delta,data)
data=flip(sortrows(data,1),1);
len=size(data,1);bre=size(data,2);
grad=0;
ZR=sum((2.^transpose(data(:,1))-1)./(log2(2:len+1)));
if(ZR==0)
    return;
end
weight=zeros(1,len);
mi=find(data(:,1)==4,1,'first');
if(isempty(mi)==0)
    ma=find(data(:,1)==4,1,'last');
    weight(mi:ma)= ((2^4-1)./log2(1+mi:1+ma))/ZR;
    val=mean(weight(mi:ma));
    weight(mi:ma)=val;
end
mi=find(data(:,1)==3,1,'first');
if(isempty(mi)==0)
    ma=find(data(:,1)==3,1,'last');
    weight(mi:ma)= ((2^3-1)./log2(1+mi:1+ma))/ZR;
    val=mean(weight(mi:ma));
    weight(mi:ma)=val;
end
mi=find(data(:,1)==2,1,'first');
if(isempty(mi)==0)
    ma=find(data(:,1)==2,1,'last');
    weight(mi:ma)= ((2^2-1)./log2(1+mi:1+ma))/ZR;
    val=mean(weight(mi:ma));
    weight(mi:ma)=val;
end
mi=find(data(:,1)==1,1,'first');
if(isempty(mi)==0)
    ma=find(data(:,1)==1,1,'last');
    weight(mi:ma)= ((2^1-1)./log2(1+mi:1+ma))/ZR;
    val=mean(weight(mi:ma));
    weight(mi:ma)=val;
end
mi=find(data(:,1)==0,1,'first');
if(isempty(mi)==0)
    ma=find(data(:,1)==0,1,'last');
    weight(mi:ma)= 0;
    val=mean(weight(mi:ma));
    weight(mi:ma)=val;
end

%Calculating gradient of phi_slam function 
for i=1:len
    vec=(data(i,1)>data(1:len,1)).*(((data(i,1)-data(1:len,1))*delta - data(i,2:bre)*wrank) +data(1:len,2:bre)*wrank);
    
    [val,index]=max(vec);

    if(val>0)
         feature=data(index,2:bre)- data(i, 2:bre);
         %temp=data(1:len,2:bre);
         %temp=bsxfun(@minus,temp,data(i,2:bre)); 
         %feature=temp(index,:);
         grad=grad+weight(i)*(feature);
    end   
         
end
end

%Gradient Function for Pure Online Algorithm
function [grad]= calculateOAgradient(wrank,data, loss)
data=flip(sortrows(data,1),1);
len=size(data,1);bre=size(data,2);
grad=0;

max_val=0;
%Calculating gradient of surrogate function 
for i=1:len
    vec=(data(i,1)>data(1:len,1)).*((1 - data(i,2:bre)*wrank) +data(1:len,2:bre)*wrank);
    [val,index]=max(vec);
        
    if(val>max_val)
        max_val=val;
        j_index=index;
        i_index=i;
    end
end

if(max_val>0)
    grad=data(j_index,2:bre)- data(i_index,2:bre);
    grad=loss*grad;     
end
end



% Calculating NDCG Loss (1- NDCG) on current query, using current ranking
% function
function[loss]= NDCGLoss(w,data)

loss=0;NDCG=zeros(10,1);

if(max(data(:,1))==0)
   return ;
end

len=size(data,1);
final=min(len,10);

% Creating a copy of document list and sorting according to
% relevance score

datatestcopy=data;
datatestcopy=flip(sortrows(datatestcopy,1),1);

% Calculating the score vector for the document list and sorting the
% documents according to score.

S=data(:,2:size(data,2))*w;
datatest=cat(2,S,data);
datatest=flip(sortrows(datatest,1),1);

%Calculating NDCG@1:final
%Calculating ZR@1:final

ZR=0;numerator=0;
for j=1:final
    ZR=ZR+ (2^datatestcopy(j,1)-1)/(log2(j+1));
    numerator=numerator + (2^datatest(j,2)-1)/(log2(j+1));
    NDCG(j)=numerator/ZR;
    
end
loss=1- NDCG(final);
end


