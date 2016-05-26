% This code is to generate the Yandex graph in perceptron paper.
% It compares performance of the two perceptron algorithms and  online ListNet
% on Yahoo dataset.

function SLAMOAListNetYahoo()
tic
% Initializing parameters
T=4;var=700;

% Stores time-cumulative and time-average NDCG@k value for the two perceptron
% algorithms and online ListNet
%(k varies according to what is set in NDCGLoss function)

CumNDCGSLAMGain=zeros(2e5,1); AvgNDCGSLAMGain=zeros(2e5,1);
CumNDCGListNetGain=zeros(2e5,1); AvgNDCGListNetGain=zeros(2e5,1);
CumNDCGOAGain=zeros(2e5,1); AvgNDCGOAGain=zeros(2e5,1);
wrankSLAM=zeros(var,1); wrankListNet= zeros(var,1); wrankOA=zeros(var,1);
count=0;

for t= 1:T
    f = fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/ltrc_yahoo/set1.train.txt');
    X = zeros(2e5,0); % Feature dimension is the column length. 
    X(1,1:700)=0;
    R = zeros(1e5,1);
    Q = zeros(1e5,1);
    qid = '';
    i = 0; q = 0;
    while 1
        l = fgetl(f);
        if ~ischar(l), break; end;
        i = i+1; 
        R(i,1)= sscanf(l,'%d',1); temprel=R(i,1);
        [lab,  foo1, foo2, ind] = sscanf(l,'%d qid:',1); l(1:ind-1)=[];
        [nqid, foo1, foo2, ind] = sscanf(l,'%s',1); l(1:ind-1)=[];Q(i,1)=str2num(nqid); tempqid= Q(i,1); 
        if ~strcmp(nqid,qid)
            if(q~=0)
                                              
                X=X(1:i-1,:);R=R(1:size(X,1),1);Q=Q(1:size(X,1),1); X=cat(2,Q,R,X);
                [gradSLAM, gradListNet, gradOA, lossSLAM, lossListNet, lossOA]= SLAMListNetBigData(X,wrankSLAM,wrankListNet,wrankOA);
                
                if(rand<0.8)
                    disp(count);
                    count=count+1;
                    % Update the rankers
                    % Update SLAM ranker
                    if(count==1)
                        CumNDCGSLAMGain(1)=1-lossSLAM;
                        AvgNDCGSLAMGain(1)=CumNDCGSLAMGain(1);
                    else
                        CumNDCGSLAMGain(count)= CumNDCGSLAMGain(count-1) + 1- lossSLAM;
                        AvgNDCGSLAMGain(count)= CumNDCGSLAMGain(count)/count;
                    end
                    wrankSLAM=wrankSLAM - 1/sqrt(count)*gradSLAM;
                
                    % Update purely online algorithm ranker
                    if(count==1)
                        CumNDCGOAGain(1)=1-lossOA;
                    AvgNDCGOAGain(1)=CumNDCGOAGain(1);
                    else
                    CumNDCGOAGain(count)= CumNDCGOAGain(count-1) + 1- lossOA;
                    AvgNDCGOAGain(count)= CumNDCGOAGain(count)/count;
                    end
                    wrankOA=wrankOA - lossOA*gradOA;
                
                    % Update ListNet ranker
                    if(count==1)
                        CumNDCGListNetGain(1)=1-lossListNet;
                        AvgNDCGListNetGain(1)=CumNDCGListNetGain(1);   
                    else
                        CumNDCGListNetGain(count)=CumNDCGListNetGain(count-1)+ 1-lossListNet;
                        AvgNDCGListNetGain(count)=CumNDCGListNetGain(count)/count;     
                    end
                    wrankListNet=wrankListNet - 1/sqrt(count)* gradListNet;
                end
                X = zeros(2e5,0); 
                R = zeros(1e5,1);
                Q = zeros(1e5,1);
                X(1,1:700)=0;
                i=1;
                R(i,1)=temprel; Q(i,1)=tempqid;
            end  
            q = q+1;
            qid = nqid;
            Y{q} = lab;
        else 
            Y{q} = [Y{q}; lab];
        end;
        tmp = sscanf(l,'%d:%f'); 
        X(i,tmp(1:2:end)) = tmp(2:2:end);
        
    end
    fclose(f);
    disp(q);
end

%Write AvgNDCG for ListNet and SLAM
fileid=fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/ListNetNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGListNetGain);
fclose(fileid);
fileid=fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/OANDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGOAGain);
fclose(fileid);
fileid=fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/SLAMNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGSLAMGain);
fclose(fileid);
% Write the rankers
fileid=fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/ListNetRanker.txt','wt');
fprintf(fileid,'%f\n',wrankListNet);
fclose(fileid);
fileid=fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/OARanker.txt','wt');
fprintf(fileid,'%f\n',wrankOA);
fclose(fileid);
fileid=fopen('/Users/sougatachaudhuri/Documents/Experiments/Learning to Rank Challenge/SLAMRanker.txt','wt');
fprintf(fileid,'%f\n',wrankSLAM);
fclose(fileid);

toc
end


% This function calculates the gradients and losses

function [gradSLAM, gradListNet, gradOA, lossSLAM, lossListNet, lossOA]= SLAMListNetBigData(X,wrankSLAM,wrankListNet,wrankOA)
gradSLAM=0;gradOA=0;delta=1;
data=X(:,2:(size(X,2)));

%SLAM-Perceptron
lossSLAM= NDCGLoss(wrankSLAM,data);
if(lossSLAM>0)
gradSLAM=transpose(calculateSLAMgradient(wrankSLAM,delta,data));
end

% Purely Online Perceptron
lossOA= NDCGLoss(wrankOA,data);
if(lossOA>0)
gradOA=transpose(calculateOAgradient(wrankOA,data));
end

%ListNet- OGD
lossListNet=NDCGLoss(wrankListNet,data);
gradListNet=calculateListNetgradient(wrankListNet,data);
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
function [grad]= calculateOAgradient(wrank,data)
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




 


      
