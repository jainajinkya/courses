function [mA,mB,mC,wts] = dynamicsTemplate(nState,nInput,nModel)
%% Model Dynamics
mA = zeros(nState,nState,nModel);
mB = zeros(nState,nState,nModel);
mC = zeros(nState,nState,nModel);

for i=1:nModel
   mA(:,:,i) = eye(nState) ;%+ rand()*eye(nState);
   mB(:,:,i) = eye(nState) ; %+ 5*rand()*eye(nState);
   mC(:,:,i) = eye(nState) ; %+ rand()*eye(nState);
end

wts = (1/nModel)*ones(nModel,1);
end