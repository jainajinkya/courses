clear variables
close all
clc

%% B_lqr Exp1
nState = 2;
nInput = 2;
nOutput = 2;
nModel = 2;
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
Q_f = 20*eye(nState);
labda = 200;


% Extende Matrices
A_ext = [[A'; zeros(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 10]' ;
F = [[Q_f'; zeros(1,nState)]';0 0 labda]';  %Terminal Cost

%% LQR Control
for t=1:t_f
    W = 0.5*(5.0-m(1))^2*eye(nOutput);
    A_ext(3,3) = s;
    
    [K,S3] = finiteLQR(t_f,A_ext,B_ext,Q_ext,R,F);
    u = -K(:,:,t)*[m;s] ;
    m = A*m + B*u + normrnd(0,s,[nState,1]);

    % Covariance Dynamics
    gamma = A*sig*A';
    dsig = sig*(A+A') - gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A'))) + ...
        gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A')*C'))*((C*gamma*C' + W)\(C*gamma))...
        - sig*(A+A')*C'*((C*gamma*C' + W)\(C*gamma));
    
    s = dsig(1,1);
    sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
    traj = [traj, m];
end
figure(1);clf;
plot(traj(1,:),traj(2,:),'r');
hold on
plot(traj(1,t_f+1),traj(2,t_f+1),'bx');
hold off

