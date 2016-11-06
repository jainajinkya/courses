clear variables
close all
clc

% %% LQG Control Exp-1
% nState = 2;
% nInput = 2;
% nOutput = 2;
% t_f = 100;
% 
% A = [1 0; 0 1];
% B = [1 0; 0 1];
% C = [1 0; 0 1];
% V = 0.02*eye(nState);
% W = 0.05*eye(nOutput);
% mu = zeros(nState);
% 
% x = [2.5; 0.0];
% y = [0.0; 0.0];
% x_hat = [2.0; 2.0];
% y = C*x + normrnd(mu,W);
% 
% 
% traj2 = [x];
% 
% Q = 0.5*eye(nState);
% R = 0.5*eye(nInput);
% F = 2*eye(nState);
% 
% K = zeros(nOutput,nOutput,t_f);
% L = zeros(nState,nState,t_f);
% S = zeros(nState,nState,t_f);
% P = zeros(nState,nState,t_f);
% S(:,:,t_f)= F;
% P(:,:,1)= zeros(nState);
% 
% for t=t_f-1:-1:1
%     S(:,:,t) = Q + A'*(S(:,:,t+1) - S(:,:,t+1)*B*((B'*S(:,:,t+1)*B + R)\(B'*S(:,:,t+1))))*A;
%     L(:,:,t) = (B'*S(:,:,t+1)*B + R)\(B'*S(:,:,t+1)*A);
% end
% 
% for t=1:t_f
% %    W = 0.5*(5-x(1))^2*eye(nOutput);
%    u = -L(:,:,t)*x_hat;
%    x = A*x + B*u + normrnd(mu(1,1),V(1,1),[nState,1]);
%    y = C*x + normrnd(mu(1,1),W(1,1),[nOutput,1]);
%   
%    if t==t_f
%       continue;
%    else
%        P(:,:,t+1) =  A*(P(:,:,t) - P(:,:,t)*C'*((C*P(:,:,t)*C' + W)\C*P(:,:,t)))*A' + V;
%        K(:,:,t+1) = P(:,:,t)*C'/(C*P(:,:,t)*C' + W);
%        x_hat = A*x_hat + B*u + K(:,:,t+1)*(y - C*(A*x_hat + B*u));
%    end
%     
%    traj2 = [traj2,x];
% end
% 
% figure(2);clf;
% scatter(traj2(1,:),traj2(2,:),'r');
% hold on
% plot(traj2(1,t_f),traj2(2,t_f),'bx');
% hold off



%% B_lqr Exp1
nState = 2;
nInput = 2;
nOutput = 2;
nCov = nState^2;
nSoln = 100;
tol = 1e-6;

t_f = 50;

x = [2.5,0]';
m = [2,2]';
sig = 5*eye(nState);
s = sig(1,1);

traj = [m];

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.1*eye(nState);
R = 1.0*eye(nInput);
Q_f = 50*eye(nState);
labda = 2000;


% Extende Matrices
A_ext = [[A'; zeros(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 10]' ;
F = [[Q_f'; zeros(1,nState)]';0 0 labda]';  %Terminal Cost

%% LQR Control
for t=1:t_f
    W = 0.5*(5.0-m(1))^2*eye(nOutput)
%     W = 0.05*eye(nOutput);
%     C = [1+(5-x(1)),0;(5-x(1)), 1];
    A_ext(3,3) = s;
    
    [K,S3] = finiteLQR(t_f,A_ext,B_ext,Q_ext,R,F);
    u = -K(:,:,t)*[m;s] ;
    m = A*m + B*u + normrnd(0,s,[nState,1]); %+ gamma*C'*((C*gamma*C' + W)\(C*gamma)); % + normrnd(0,s,[nState,1]);

    % Covariance Dynamics
    gamma = A*sig*A';
    dsig = sig*(A+A') - gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A'))) + ...
        gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A')*C'))*((C*gamma*C' + W)\(C*gamma))...
        - sig*(A+A')*C'*((C*gamma*C' + W)\(C*gamma));
    
    s = dsig(1,1);
    sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
    traj = [traj, m];
end
s
figure(1);clf;
plot(traj(1,:),traj(2,:),'r');
hold on
plot(traj(1,t_f+1),traj(2,t_f+1),'bx');
hold off

