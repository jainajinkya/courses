function [muNew,covNew] = modelDynamics(x,mu,cov,u,mA,mB,mC,wts)
global nState nModel nGauss

muSet = zeros(nState,nGauss,nModel);
covSet = zeros(nState,nState,nGauss,nModel);

% Inclusion of covariance in the formulation
%W = 0.5*(5.0-x(1))^2*eye(nState);
W = 0.5*eye(nState);

for k=1:nGauss
    for i=1:nModel
        gamma = mA(:,:,i)*cov(:,:,k)*mA(:,:,i)';        
        muSet(:,k,i) = mA(:,:,i)*mu(:,k) + mB(:,:,i)*u;
        covSet(:,:,k,i) = gamma - gamma*mC(:,:,i)'*(inv(mC(:,:,i)*gamma*mC(:,:,i)' + W))*(mC(:,:,i)*gamma);
    end
end

[muNew,covNew] = genGMM(muSet,covSet,wts);
end
