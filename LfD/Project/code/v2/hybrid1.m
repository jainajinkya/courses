clear
close all
clc

%% B_lqr Exp1
nState = 2;
nInput = 2;
nOutput = 2;

t_f = 100;
t_res = 1;

x = [2.5,0]';
m = [2,2]';
sig = 5*eye(nState);

traj = [m];
traj_wo_err = [m];
U = [u];

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];

% Cost Matrices
Q = 0.01*eye(nState);
R = 0.5*eye(nInput);
Q_f = 20*eye(nState);
labda = 2000;


%% Model Dynamics
nModel = 3;
nGauss = 5;

mu = zeros(nState,nGauss);
cov = zeros(nState,nState,nGauss);
mProb = zeros(nModel,1);

mA = zeros(nState,nState,nModel);
mB = zeros(nState,nState,nModel);
mA_ext = zeros(nState+1,nState+1,nModel);
mB_ext = zeros(nState+1,nState+1,nModel);

for i=1:nGauss
   mu(:,i) = [2,2]' + normrnd(0,0.1,[nState,1]);  
   cov(:,:,i) = sig + normrnd(0,0.1)*eye(nState);
end

for i=1:nModel
   mA(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
   mB(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
   mA_ext = [[mA(:,:,i)'; zeros(1,nState)]';0 0 0]';
   mB_ext = [mB(:,:,i); 0 0];
end


























