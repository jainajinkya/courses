function [muNew, covNew] = genGMM(mu,cov,wts)
% Right now we are just taking average of all the gmm means and covariance.
% This should be tested against the case when you are fitting a new K mode 
% GMM on the new distribution of the means
nState = size(mu,1);
nGauss = size(mu,2);
nModel = size(mu,3);

muNew = zeros(nState,nGauss);
covNew = zeros(nState,nState,nGauss);

for i=1:nState
    for j=1:nGauss
        for k=1:nModel
            muNew(i,j) = muNew(i,j) + wts(k)*mu(i,j,k);
            covNew(i,:,j) = covNew(i,:,j) + wts(k)*cov(i,:,j,k);
        end
    end
end
% X=[];
% for i=1:nGauss
%    X = [X;mvnrnd(muNew(1,i),cov(1,1,i),1000)] ;
% end
% gm = fitgmdist(X,nGauss);
% gm.mu
%     
% X=[];
% for i=1:nGauss
%    X = [X;mvnrnd(muNew(2,i),cov(2,2,i),1000)] ;
% end
% gm = fitgmdist(X,nGauss);
% gm.mu
%     
end