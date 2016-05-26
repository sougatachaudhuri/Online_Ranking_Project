% Online learning to rank with feedback at the top
% This is the code for the algorithm RTop-kF in Online learning to rank paper.
% The code applies RTop-kF with 4 surrogates: Squared (k=1), KL (k=1),
% SmoothDCG (k=1) and RankSVM (k=2). KL is the un-normalized ListNet.
% The code also includes ListNet algorithm as the full information
% algorithm and purely random algorithm as no information algorithm.
% Tuning of the algorithmic parameters are of vital importance, but I am
% not putting too much emphasis on that for now.

% Main function
function RankingTopKF()
tic
% Initializing parameters
totalruns=1; % Number of times the code will be repeated, to account for the randomness of the RTop-kF algorithm. Ideally, totalruns ~ 10 or more, to 
% minimize the effect of randomness, but it will take a lot of time on Yahoo dataset.
T=10;var=700; % T=2 is is the number of times the code runs over the entire Yahoo dataset. Var is the feature dimension of Yahoo matrices
U=1; %Radius of ball containing ranking parameters

%eta=0.001; gamma= 0.01; % Learning and mixing parameter

CumNDCGSquaredGain=zeros(4e5,1); AvgNDCGSquaredGain=zeros(4e5,1);
CumNDCGSvmGain=zeros(4e5,1); AvgNDCGSvmGain=zeros(4e5,1);
CumNDCGKLGain=zeros(4e5,1); AvgNDCGKLGain=zeros(4e5,1);
CumNDCGSmoothGain=zeros(4e5,1); AvgNDCGSmoothGain=zeros(4e5,1);
CumNDCGRandomGain=zeros(4e5,1); AvgNDCGRandomGain=zeros(4e5,1);
CumNDCGListNetGain=zeros(4e5,1); AvgNDCGListNetGain=zeros(4e5,1);
wrankSquared=zeros(var,1); wrankSvm=zeros(var,1); wrankKL=zeros(var,1); wrankSmooth=zeros(var,1); wrankListNet= zeros(var,1);

% The whole experiment needs to be repeated a few times with average of the
% returned values as final output, since this is a randomized algorithm.
for runs=1:totalruns
    count=0;
    for t= 1:T
        f = fopen('~/Private/LearningtoRankChallenge/set1.train.txt');
        X = zeros(2e5,0); % Feature dimension is the column length. 
        X(1,1:700)=0;
        R = zeros(1e5,1);
        Q = zeros(1e5,1);
        qid = '';
        i = 0; q = 0;
        while 1
            % Reading a query and associated document matrix and relevance
            % vector.
	    
            l = fgetl(f);
            if ~ischar(l), break; end;
            i = i+1; 
            R(i,1)= sscanf(l,'%d',1); temprel=R(i,1);
            [~,  ~, ~, ind] = sscanf(l,'%d qid:',1); l(1:ind-1)=[];
            [nqid, ~, ~, ind] = sscanf(l,'%s',1); l(1:ind-1)=[];Q(i,1)=str2num(nqid); tempqid= Q(i,1); 
            % Finished reading
            if ~strcmp(nqid,qid)
                if(q~=0)
                    %if (rand<0.5) % Will allow the selected query to be processed with probability 0.5, so as to avoid repetition of seeing queries in order over every run
                        count=count+1;
                        disp(count);
                        X=X(1:i-1,:);R=R(1:size(X,1),1);Q=Q(1:size(X,1),1); X=cat(2,Q,R,X);
                        gamma=0.1/(nthroot(count,3)); % One of the RTop-1F parameters. Using the theoretically derived value, but more tuning can be done here.
                        [gradSquared, lossSquared, gradSvm, lossSvm, gradKL, lossKL, gradSmooth, lossSmooth, lossRandom, gradListNet, lossListNet]= LossAndGradient(X,wrankSquared, wrankSvm, wrankKL, wrankSmooth, wrankListNet, gamma, count); 
                        % Update the rankers   
                        eta=0.1/(nthroot(count^2,3)); % Another RTop-1F parameter,using theoretically derived value, but can be tuned.
                        if(count==1)
                            CumNDCGSquaredGain(1)=1-lossSquared;
                        else
                            CumNDCGSquaredGain(count)= CumNDCGSquaredGain(count-1) + 1- lossSquared;
                        end
                        wranktemp=wrankSquared - (eta)*gradSquared;
                        wrankSquared= min(1, U/norm(wranktemp))*wranktemp; 
                    
                        if(count==1)
                            CumNDCGSvmGain(1)=1-lossSvm;
                        else
                            CumNDCGSvmGain(count)= CumNDCGSvmGain(count-1) + 1- lossSvm;
                        end
                        wranktemp=wrankSvm - (eta)*gradSvm;                    
                        wrankSvm= min(1, U/norm(wranktemp))*wranktemp;  
                    
                        if(count==1)
                            CumNDCGKLGain(1)=1-lossKL;
                        else
                            CumNDCGKLGain(count)= CumNDCGKLGain(count-1) + 1- lossKL;
                        end
                        wranktemp=wrankKL - (eta)*gradKL;                    
                        wrankKL= min(1, U/norm(wranktemp))*wranktemp;     
                    
                        if(count==1)
                            CumNDCGSmoothGain(1)=1-lossSmooth;
                        else
                            CumNDCGSmoothGain(count)= CumNDCGSmoothGain(count-1) + 1- lossSmooth;
                        end
                        wranktemp=wrankSmooth - (eta)*gradSmooth;
                        wrankSmooth= min(1, U/norm(wranktemp))*wranktemp;
                    
                        if(count==1)
                            CumNDCGListNetGain(1)=1-lossListNet;                     
                        else
                            CumNDCGListNetGain(count)=CumNDCGListNetGain(count-1)+ 1-lossListNet;                        
                        end
                        wranktemp=wrankListNet - (0.1/sqrt(count))*gradListNet; % 0.1/sqrt(count) is the ListNet learning parameter. Can be tuned.
                        wrankListNet= min(1, U/norm(wranktemp))*wranktemp;
                    
                        if(count==1)
                            CumNDCGRandomGain(1)=1-lossRandom;
                        else
                            CumNDCGRandomGain(count)= CumNDCGRandomGain(count-1) + 1- lossRandom;
                        end                
                    %end
                    X = zeros(2e5,0); 
                    R = zeros(1e5,1);
                    Q = zeros(1e5,1);
                    X(1,1:700)=0;
                    i=1;
                    R(i,1)=temprel; Q(i,1)=tempqid;
                end  
                q = q+1;
                qid = nqid;              
            
            end
            tmp = sscanf(l,'%d:%f'); 
            X(i,tmp(1:2:end)) = tmp(2:2:end);
        end
        fclose(f);
        
    end
    
    if (runs==1)
        newcount=count;
        AvgNDCGSquaredGain=CumNDCGSquaredGain;
        AvgNDCGSvmGain=CumNDCGSvmGain;
        AvgNDCGKLGain=CumNDCGKLGain;
        AvgNDCGSmoothGain=CumNDCGSmoothGain;
        AvgNDCGListNetGain=CumNDCGListNetGain;
        AvgNDCGRandomGain=CumNDCGRandomGain;
        finalcount=count;
    else
        if(count<newcount)
	  newcount=count;
        end
        AvgNDCGSquaredGain(1:newcount)= AvgNDCGSquaredGain(1:newcount)+ CumNDCGSquaredGain(1:newcount);
        AvgNDCGSvmGain(1:newcount)= AvgNDCGSvmGain(1:newcount)+ CumNDCGSvmGain(1:newcount);
        AvgNDCGKLGain(1:newcount)= AvgNDCGKLGain(1:newcount)+ CumNDCGKLGain(1:newcount);
AvgNDCGSmoothGain(1:newcount)= AvgNDCGSmoothGain(1:newcount)+ CumNDCGSmoothGain(1:newcount);
        AvgNDCGListNetGain(1:newcount)= AvgNDCGListNetGain(1:newcount) + CumNDCGListNetGain(1:newcount);
        AvgNDCGRandomGain(1:newcount)= AvgNDCGRandomGain(1:newcount)+ CumNDCGRandomGain(1:newcount); 
        
    end
    
end

%Time averaging over the cumulated NDCG
for i=1:newcount
AvgNDCGSquaredGain(i)= AvgNDCGSquaredGain(i)/(i*totalruns);
AvgNDCGSvmGain(i)= AvgNDCGSvmGain(i)/(i*totalruns);
AvgNDCGKLGain(i)= AvgNDCGKLGain(i)/(i*totalruns);
AvgNDCGSmoothGain(i)= AvgNDCGSmoothGain(i)/(i*totalruns);
AvgNDCGListNetGain(i)= AvgNDCGListNetGain(i)/(i*totalruns);
AvgNDCGRandomGain(i)= AvgNDCGRandomGain(i)/(i*totalruns);
end


%Write AvgNDCG for ListNet and SLAM
fileid=fopen('~/Private/LearningtoRankChallenge/ResultsTopK/Param1_SquaredNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGSquaredGain);
fclose(fileid);
fileid=fopen('~/Private/LearningtoRankChallenge/ResultsTopK/Param1_SvmNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGSvmGain);
fclose(fileid);
fileid=fopen('~/Private/LearningtoRankChallenge/ResultsTopK/Param1_KLNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGKLGain);
fclose(fileid);
fileid=fopen('~/Private/LearningtoRankChallenge/ResultsTopK/Param1_SmoothNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGSmoothGain);
fclose(fileid);
fileid=fopen('~/Private/LearningtoRankChallenge/ResultsTopK/Param1_ListNetNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGListNetGain);
fclose(fileid);
fileid=fopen('~/Private/LearningtoRankChallenge/ResultsTopK/Param1_RandomNDCG10.txt','wt');
fprintf(fileid,'%f\t\n',AvgNDCGRandomGain);
fclose(fileid);

toc
end


% This function calculates the gradients and losses
function [gradSquared, lossSquared, gradSvm, lossSvm, gradKL, lossKL, gradSmooth, lossSmooth, lossRandom, gradListNet, lossListNet]= LossAndGradient(X,wrankSquared, wrankSvm, wrankKL, wrankSmooth,wrankListNet, gamma, count)
data=X(:,2:(size(X,2)));len=size(data,1);
gradSquared=0;lossSquared=0;gradSvm=0;lossSvm=0; gradKL=0;lossKL=0;
gradSmooth=0;lossSmooth=0;lossRandom=0;gradListNet=0;lossListNet=0;
if (len<2)
    return
end
% Getting feedback on top-2 elements according to permutation selected by
% algorithm
[R_1_Squared, R_2_Squared, sigma_Squared, flag_11, flag_21]= Top2Feedback(wrankSquared,data, gamma);
[R_1_Svm, R_2_Svm, sigma_Svm, flag_12, flag_22]= Top2Feedback(wrankSvm,data, gamma);
[R_1_KL, R_2_KL, sigma_KL, flag_13, flag_23]= Top2Feedback(wrankKL,data, gamma);

[R_1_Smooth, R_2_Smooth, sigma_Smooth, flag_14, flag_24]= Top2Feedback(wrankSmooth,data, gamma);

% Calculate the losses
lossSquared= NDCGLossRandom(sigma_Squared, data);
lossSvm= NDCGLossRandom(sigma_Svm, data);
lossKL= NDCGLossRandom(sigma_KL, data);
lossSmooth= NDCGLossRandom(sigma_Smooth, data);
lossListNet=NDCGLoss(wrankListNet,data);
randomsigma=randperm(len);
lossRandom= NDCGLossRandom(randomsigma,data);

% Construction of unbiased estimators of the gradients
gradSquared= unbiasedestimatorSquared(data, wrankSquared, R_1_Squared,sigma_Squared,gamma, flag_11);
gradSvm= unbiasedestimatorSvm(data, wrankSvm, R_1_Svm, R_2_Svm, sigma_Svm, gamma, flag_22);
gradKL= unbiasedestimatorKL(data, wrankKL, R_1_KL,sigma_KL,gamma, flag_13);
gradSmooth= unbiasedestimatorSmooth(data, wrankSmooth, R_1_Smooth,sigma_Smooth,gamma, flag_14);
gradListNet=calculateListNetgradient(wrankListNet,data);

end

function [R_1,R_2, sigma, flag_1, flag_2]= Top2Feedback(w,data, gamma)
len=size(data,1);
temp=transpose(1:len);
scores=data(:,2:size(data,2))*w;
temp=cat(2,scores,temp);
sigma_hat=flip(sortrows(temp,1),1);
sigma_hat=sigma_hat(:,2);
% Randomizing over the permutations
rand_point=unifrnd(0,1);
if(rand_point>(1-gamma)) %Ignoring gamma/m!
    % Select a random permutation uniformly
    sigma=randperm(len); 
   
else
    % Select sigma_hat
    sigma= sigma_hat;  
   
end
flag_1=0; flag_2=0;
if (sigma(1)==sigma_hat(1) && sigma(2)==sigma_hat(2))
    flag_2=1;
end
if (sigma(1)==sigma_hat(1))
    flag_1=1;
end
% Get feedback on top-2 elements of sigma
R_1= data(sigma(1),1);R_2=data(sigma(2),1);
end

% Function to construct unbiased estimator of gradient of squared loss
function[grad]= unbiasedestimatorSquared(data, wrank, R_1,sigma,gamma, flag)
len=size(data,1);
features=data(:,2:size(data,2));
score= features*wrank;
coordvector= zeros(len,1);
coordvector(sigma(1))= R_1;
if (flag==1)
    denom=1-gamma + gamma/len;
else
    denom=0.001*gamma/len;
end
coordvector=coordvector/denom;
grad=features'*(2*(score- coordvector));
end

% Function to construct unbiased estimator of gradient of RankSvm surrogate
function[grad]= unbiasedestimatorSvm(data, wrank, R_1, R_2,sigma,gamma,flag)
len=size(data,1); bre=size(data,2)-1; grad=zeros(bre,1);
coordvector=zeros(len,1);
features=data(:,2:size(data,2));
score= features*wrank;
if(R_1>R_2)
   if(1 + score(sigma(2))> score(sigma(1)))
       coordvector(sigma(2))=1;
       coordvector(sigma(1))=-1;
       grad= features'* coordvector;
   end
elseif(R_2>R_1)
    if(1+ score(sigma(1))> score(sigma(2)))
        coordvector(sigma(2))=-1;
        coordvector(sigma(1))=1;
        grad=features'*coordvector;
    end
 
end
if (flag==1)
    denom=1-gamma + gamma/(len*(len-1));
else
    denom=0.001*gamma/(len*(len-1));
end

grad=grad/denom;

end

% Function to construct unbiased estimator of gradient of KL surrogate
function[grad]= unbiasedestimatorKL(data, wrank, R_1,sigma,gamma, flag)
len=size(data,1);
features=data(:,2:size(data,2));
score= features*wrank;
coordvector= zeros(len,1);
temp=exp(score(sigma(1)))- exp(R_1);
if (flag==1)
    denom=1-gamma + gamma/len;
else
    denom=0.001*gamma/len;
end

coordvector(sigma(1))= temp/denom;
grad=features'*coordvector;

end

% Function to construct unbiased estimator of gradient of KL surrogate
function[grad]= unbiasedestimatorSmooth(data, wrank, R_1,sigma,gamma, flag)
epsilon=0.001; % Parameter specific to SmoothDCG function
len=size(data,1);
features=data(:,2:size(data,2));
score= features*wrank;
denom=sum(exp(score/epsilon));
coordvector= -1*(exp(score(sigma(1))/epsilon)* exp(score/epsilon))/(epsilon*(denom)^2);
coordvector(sigma(1))= coordvector(sigma(1)) + exp(score(sigma(1))/epsilon)/(epsilon* denom);
if (flag==1)
    denom=1-gamma + gamma/len;
else
    denom=0.001*gamma/len;
end

coordvector=coordvector*((2^R_1-1)/denom);
grad= features'* coordvector;
end

% Gradient Function for ListNet Loss
function [grad]=calculateListNetgradient(wrank,data)
% Calculating the gradient from Eq.6 of ListNet paper.
data=transpose(data); len=size(data,1); 
global datatranspose;
datatranspose=data;
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


% Calculating NDCG Loss (1- NDCG) on current query, using given ranking
function[loss]= NDCGLossRandom(sigma,data)
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

% Sorting the documents according to sigma
datatest=data(sigma,:);
%Calculating NDCG@1:final

ZR=0;numerator=0;
for j=1:final
    ZR=ZR+ (2^datatestcopy(j,1)-1)/(log2(j+1));
    numerator=numerator + (2^datatest(j,1)-1)/(log2(j+1));
    NDCG(j)=numerator/ZR;
    
end
loss=1- NDCG(final);

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
