% Code for plotting regret, with top-k feedback and various block sizes

function DCGRegret()
T=10000;m=20;p1=0.7; p2=0.2;
% Generating relevance vectors
adversaryrel1=binornd(1,p1,m/2,T);
adversaryrel2=binornd(1,p2,m/2,T);
adversaryrel=vertcat(adversaryrel1, adversaryrel2);


%Creating constants for RT-1F algorithm
K=100;blocksize=100;eta=40;totalruns=10;k=1; % T= K*blocksize, eta=1/epsilon

num=1:1:m;num=transpose(num);SumLossFinal=zeros(T,1);SumLossFullInfo=zeros(T,1);
% An outer loop will simulate the expectation over the relevance vectors.
% Inner loop makes one pass over T relevance vectors.
% sigma will indicate permutation from objects to ranks. Thus, sigma(1)=2
% means object 1 has been placed at position 2.
% Creating the ceil(m/k) partition

% Loss under partial information
for runs=1:totalruns
    cumloss=0;count=0;estimatedrel=zeros(m,1);blockrel=zeros(m,1);loss=zeros(T,1);
    for i=1:K
        
        estimatedrel=estimatedrel+blockrel;
        %Sample ceil(m/k) uniform time points in the block B_k
        tempdata=((i-1)*blocksize+1):1:min((i*blocksize),T); 
        explorepoints=randsample(tempdata,ceil(m/k));
        for t= (i-1)* blocksize +1: min((i*blocksize),T)
            count=count+1; 
            idx=find(explorepoints(:)==t);
            if(isempty(idx)==0)
               
               % Find permutation which places objects in idx cell at top positions and then calculate the loss
               objects_at_top= k*(idx-1)+1:1: min(k*idx,m);
               sigma=randperm(m);
               for j=1:length(objects_at_top)
                   %swap
                   temp=sigma(objects_at_top(j));
                   idx_1= find(sigma(:)==j);
                   sigma(objects_at_top(j))=j;
                   sigma(idx_1)=temp;
               end
                              
               % Receive actual relevance of top placed objects
               blockrel(objects_at_top(:))= adversaryrel(objects_at_top(:),count);
               % Calculate loss in the round
               dcgvector=1./(log2(1+sigma));
               cumloss=cumloss+dcgvector*adversaryrel(:,count);
               loss(t)=cumloss;
            else
               % Output permutation according to Kalai-Vempala Algorithm
               p=unifrnd(0,eta,m,1);
               temprel=estimatedrel+p;
               temprel1=[temprel num];
               temprel1= flip(sortrows(temprel1,1),1);
               sigma(temprel1(:,2))=num;
               % Calculate loss in the round
               dcgvector=1./(log2(1+sigma));
               cumloss=cumloss+dcgvector*adversaryrel(:,count);
               loss(t)=cumloss;
            end
        end  
       % Dont need to put blockrel in a vector as it is already formed in
       % vector form
    end
    SumLossFinal=SumLossFinal + loss;
end
SumLossFinal=SumLossFinal/totalruns;

%Loss under full information
totalrel=zeros(m,1);
for t=1:T
    p=unifrnd(0,eta,m,1);
    totalreltemp=totalrel+p;
    totalrel1=[totalreltemp num];
    totalrel1= flip(sortrows(totalrel1,1),1);
    sigma(totalrel1(:,2))=num;
    dcgvector=1./(log2(1+sigma));
    cumloss=cumloss+dcgvector*adversaryrel(:,t);
    SumLossFullInfo(t)=cumloss;
    totalrel=totalrel + adversaryrel(:,t);
end


%Calculating best permutation in hindsight
MaxLossFinal=zeros(T,1);
cumadversaryrel=sum(adversaryrel(:,:),2);
temprel=[cumadversaryrel num];
temprel= flip(sortrows(temprel,1),1);
sigma(temprel(:,2))=num;cumloss=0;
for t=1:T
    dcgvector=1./(log2(1+sigma));
    cumloss=cumloss+dcgvector*adversaryrel(:,t);
    MaxLossFinal(t)=cumloss;
end

%Calculating Regret
time=1:1:T;
time=transpose(time);

regretpartial=(MaxLossFinal- SumLossFinal)./time;
regretfull=(MaxLossFinal-SumLossFullInfo)./time;

fileid=fopen('/Users/sougatachaudhuri/Box Sync/Programs/OnlineRankingTop-1(NC)/DCG100.txt','wt');
fprintf(fileid,'%f\t\n',regretpartial);
fclose(fileid);
fileid=fopen('/Users/sougatachaudhuri/Box Sync/Programs/OnlineRankingTop-1(NC)/DCG250-Full.txt','wt');
fprintf(fileid,'%f\t\n',regretfull);
fclose(fileid);

%regretfull=(MaxLossFinal-SumLossFullInfo)./time;
% disp(regretpartial(end-10:end));
% % Plots
% xrange=10:1:T;
% yrange=regretpartial(10:T);
% hand=plot(xrange,yrange, 'r');
% set(gca, 'fontsize',20);
% set(hand,'LineWidth',2);
% xlabel('Iterations');
% ylabel('AverageRegret');
% hleg = legend('AvgRegret@Top-1');
% set(hleg, 'FontSize',20);
% figure();
% xrange=1000:90:T;
% yrange=regretpartial(xrange);
% %yrange=regretpartial(1000:T);
% temp=regretfull(1000);
% regretfull(1000:T)=regretfull(1000:T)*regretpartial(1000)/temp;
% yrange1=regretfull(xrange);
% hand=plot(xrange,yrange,'-*',xrange,yrange1,'-o');
% set(gca, 'fontsize',20);
% set(hand,'LineWidth',2);
% xlabel('Iterations');
% ylabel('AverageRegret');
% hleg = legend('AvgRegret@Top-1','AvgRegret@FullInfo');
% set(hleg, 'FontSize',20);
end




