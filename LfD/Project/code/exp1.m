clear
close all
clc

%% B_lqr Exp1
nState = 2;
nInput = 2;
nOutput = 2;
nCov = nState^2;
nSoln = 100;
tol = 1e-6;

t_f = 100;
t_res = 0.5;

x = [2.5,0]';
m = [2,2]';
sig = 5*eye(nState);
s = sig(1,1);

traj = [m];

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
Q_f = 50*eye(nState);
u_max = 0.1;
labda = 200;


% Extende Matrices
A_ext = [[A'; zeros(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 10]' ;
F = [[Q_f'; zeros(1,nState)]';0 0 labda]';  %Terminal Cost

%% LQR Control
count = 1;
for t=1:t_res:t_f
   W = 0.5*(5.0-m(1))^2*eye(nOutput);
%     W = 0.05*eye(nOutput);
%     C = [1+(5-x(1)),0;(5-x(1)), 1];
    A_ext(3,3) = s;
    
    [K,S3] = finiteLQR(t_f,A_ext,B_ext,Q_ext,R,F,t_res);
    u = -K(:,:,count)*[m;s];
    
    for j=1:nInput
        if abs(u(j)) > u_max
            u(j) = (u(j)/abs(u(j)))*u_max;
        end
    end
    
    m = A*m + B*u + normrnd(0,s,[nState,1]);
    
    % Covariance Dynamics
    gamma = A*sig*A';
    dsig = sig*(A+A') - gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A'))) + ...
        gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A')*C'))*((C*gamma*C' + W)\(C*gamma))...
        - sig*(A+A')*C'*((C*gamma*C' + W)\(C*gamma));
    
    s = dsig(1,1);
    sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
    traj = [traj, m];
    count = count+1;
end

figure(3);
hold on
plot(traj(1,:),traj(2,:),'r');
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
hold off
save('traj_data5.mat', 'traj');
