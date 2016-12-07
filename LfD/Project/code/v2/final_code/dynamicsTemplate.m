function [mA,mB,mC,wts] = dynamicsTemplate()
global nState nModel nInput nOutput

pitch = 0.1;
%% Model Dynamics
mA = zeros(nState,nState,nModel);
mB = zeros(nState,nState,nModel);
mC = zeros(nState,nState,nModel);

for i=1:nModel
    if(i==0)
            A = eye(nState);
            B = eye(nInput);
            C = eye(nOutput);
            
    elseif(i==1)
            A = eye(nState);
%             B = zeros(nState,nInput);
            B = eye(nState);
            C = eye(nOutput);
            
%             B(1,1) = 8;
%             B(2,2) = 1;
            
    elseif(i==2)
            A = eye(nState);
            B = 1.0*eye(nInput);
            C = eye(nOutput);      
    end    
    
   mA(:,:,i) = A;
   mB(:,:,i) = B;
   mC(:,:,i) = C;
end

wts = (1/nModel)*ones(nModel,1);
end