function [K,S] = finiteLQR(tf,A,B,Q,R,F)
nState = size(A,1);
nInput = size(R,1);
S = zeros(nState,nState,tf);
S(:,:,tf)= F;
K = zeros(nInput,nState,tf);

for t=tf-1:-1:1
    % Backward Pass
    S(:,:,t)= Q + A'*S(:,:,t+1)*A - A'*S(:,:,t+1)*B*((B'*S(:,:,t+1)*B+ R)\(B'*S(:,:,t+1)*A));
end

for t= 1:tf
    K(:,:,t) = (B'*S(:,:,t)*B + R)\(B'*S(:,:,t)*A);
end
end

