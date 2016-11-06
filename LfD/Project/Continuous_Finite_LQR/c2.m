clear
close all
clc

% nState = 2;
% nInput = 2;
% 
% x = [4.0; 2.5];
% x2 = x;
% t_f = 50;
% 
% traj = [x];
% traj2 = [x2];
% 
% A = [1 0; 0 1];
% B = [1 0; 0 1];
% C = [1 0; 0 1];
% Q = 0.5*eye(nState);
% R = 0.5*eye(nInput);
% F = 200*eye(nState);  %Terminal Cost
% 
% %% LQR Control
% 
% S = zeros(2,2,t_f);
% S(:,:,t_f)= Q;
% 
% % for t=t_f-1:-1:1
% %     % Backward Pass
% %     S(:,:,t)= Q + A'*S(:,:,t+1)*A - A'*S(:,:,t+1)*B*((B'*S(:,:,t+1)*B+ R)\(B'*S(:,:,t+1)*A));
% % end
% [K,S] = finiteLQR(t_f,A,B,Q,R,F);
% 
% for t= 1:t_f
% %     u = -(B'*S(:,:,t)*B + R)\(B'*S(:,:,t)*A)*x;
%     u = -K(:,:,t)*x;
% %     [K,S2,E] = lqr(A,B,Q,R);
% %     u2 = -K*x
%     
%     % Dynamics
%     x = A*x + B*u;
% %     x2 = A*x2 + B*u2;
% 
%     traj = [traj,x];
% %     traj2 = [traj2,x2];
%     
% end
% 
% figure(1);clf;
% scatter(traj(1,:),traj(2,:),'r');
% hold on
% scatter(traj2(1,:),traj2(2,:),'b');
% hold off

%% Basic LQG Control
nState = 2;
nInput = 2;
nOutput = 2;
t_f = 100;

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
V = 0.01*eye(nState);
W = 0.05*eye(nOutput);
mu = zeros(nState);

x = [4.0; 2.5];
y = [0.0; 0.0];
x_hat = [2.0; 3.5];
y = C*x + normrnd(mu,W);

traj2 = [x];

Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
F = 200*eye(nState);

K = zeros(nOutput,nOutput,t_f);
L = zeros(nState,nState,t_f);
S = zeros(nState,nState,t_f);
P = zeros(nState,nState,t_f);
S(:,:,t_f)= F;
P(:,:,1)= zeros(nState);

for t=t_f-1:-1:1
    S(:,:,t) = Q + A'*(S(:,:,t+1) - S(:,:,t+1)*B*((B'*S(:,:,t+1)*B + R)\(B'*S(:,:,t+1))))*A;
    L(:,:,t) = (B'*S(:,:,t+1)*B + R)\(B'*S(:,:,t+1)*A);
end

for t=2:t_f
   P(:,:,t) =  A*(P(:,:,t-1) - P(:,:,t-1)*C'*((C*P(:,:,t-1)*C' + W)\C*P(:,:,t-1)))*A' + V;
   K(:,:,t) = P(:,:,t-1)*C'/(C*P(:,:,t-1)*C' + W);
end

%Dynamics

for t=1:t_f
   u = -L(:,:,t)*x_hat;
   x = A*x + B*u + normrnd(mu,V);
   y = C*x + normrnd(mu,W);
   
   if t==t_f
      continue;
   else
       x_hat = A*x_hat + B*u + K(:,:,t+1)*(y - C*(A*x_hat + B*u));
   end
   traj2 = [traj2,x];
end
figure(2);clf;
scatter(traj2(1,:),traj2(2,:),'r');