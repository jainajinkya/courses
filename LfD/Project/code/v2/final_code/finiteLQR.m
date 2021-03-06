function [K,S] = finiteLQR(t_f,A,B,Q,R,F)
nState = size(A,1);
nInput = size(R,1);
S = zeros(nState,nState,t_f);
S(:,:,t_f)= F;
K = zeros(nInput,nState,t_f);

for t=t_f-1:-1:1
    % Backward Pass
    S(:,:,t)= Q + A'*S(:,:,t+1)*A - A'*S(:,:,t+1)*B*(inv(B'*S(:,:,t+1)*B+ R)*(B'*S(:,:,t+1)*A));
end

for t= 1:t_f
    K(:,:,t) = inv(B'*S(:,:,t)*B + R)*(B'*S(:,:,t)*A);
end
end

