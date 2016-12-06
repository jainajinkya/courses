function [mA,mB,mC,wts] = dynamicsTemplate()
global nState nModel nInput nOutput

pitch = 0.1;
%% Model Dynamics
mA = zeros(nState,nState,nModel);
mB = zeros(nState,nState,nModel);
mC = zeros(nState,nState,nModel);

for i=1:nModel
    if(i==3)
            A = eye(nState);
            B = eye(nInput);
            C = eye(nOutput);
            
    elseif(i==2)
            A = eye(nState);
            B = zeros(nState,nInput);
            C = eye(nOutput);
            
            B(3,6) = (pitch/(2*pi));
            B(6,6) = 1;
            
    elseif(i==1)
            A = eye(nState);
            B = zeros(nState,nInput);
            C = eye(nOutput);      
    end    
    
   mA(:,:,i) = A;
   mB(:,:,i) = B;
   mC(:,:,i) = C;
end

wts = (1/nModel)*ones(nModel,1);
end