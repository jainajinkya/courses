function [muNew, covNew] = genGMM(mu,cov,wts)
% Right now we are just taking average of all the gmm means and covariance.
% This should be tested against the case when you are fitting a new K mode 
% GMM on the new distribution of the means
nDim = size(mu,1);
nGauss = size(mu,2);
nModel = size(mu,3);

muNew = zeros(nState,nGauss);
covNew = zeros(nState,nState,nGauss);

for i=1:nDim
    for j=1:nGauss
        for k=1:nModel
            muNew(i,j) = muNew(i,j) + wts(k)*mu(i,j,k);
            covNew(i,:,j) = covNew(i,:,j) + wts(k)*cov(i,:,j,k);
        end
    end
end


end