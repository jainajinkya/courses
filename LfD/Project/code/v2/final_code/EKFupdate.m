function [muNew,sNew] = EKFupdate(x,mu,s,u,mA,mB,mC,wts)
global nState nGauss nModel

cov = s*eye(nState);
muSet = zeros(nState,nGauss,nModel);
covSet = zeros(nState,nState,nGauss,nModel);
% W = 0.5*eye(nState);
W = 0.5*(5.0-x(1))^2*eye(nState);

for k=1:nGauss
    for i=1:nModel
        gamma = mA(:,:,i)*cov(:,:,k)*mA(:,:,i)';        
        muSet(:,k,i) = mA(:,:,i)*mu(:,k) + mB(:,:,i)*u;
        % Observation
        zTrue = mC(:,:,i)*(x-muSet(:,k)) + mC(:,:,i)*(muSet(:,k)) + mvnrnd(zeros(nState,1),W)';
        zModel = obsFn(muSet(:,k,i),W);
        muSet(:,k,i) = muSet(:,k,i) + (gamma*mC(:,:,i)'*(inv(mC(:,:,i)*gamma*mC(:,:,i)' + W))*(zTrue-zModel));
        covSet(:,:,k,i) = gamma - gamma*mC(:,:,i)'*(inv(mC(:,:,i)*gamma*mC(:,:,i)' + W))*(mC(:,:,i)*gamma);
    end
end
[muNew,covNew] = genGMM(muSet,covSet,wts);
sNew = covNew(1,1,:);
end