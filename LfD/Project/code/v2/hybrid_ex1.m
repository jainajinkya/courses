clearvars
close all
clc

global nState nInput nOutput nModel nGauss T goal Q R Q_f labda mA mB mC chpts
nState = 2;
nInput = 2;
nOutput = 2;
nModel = 3; % Check number of changepoints
nGauss = 1;
goal = [0,0]';
T = 8; 

% Dynamics Matrics
[mA,mB,mC] = dynamicsTemplate();

% Changepoints
chpts = [[3,0]',[5,2]'];  % Number of chnagepoints should be dependent on nModels

% Cost Matrices
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
Q_f = 20*eye(nState);
labda = 200;

theta = 0.1;

% System Dynamics
x0 = [2.5,0]';

% GMM Intialize
mu0 = zeros(nState,nGauss);
cov0 = zeros(nState,nState,nGauss);
% mProb = zeros(nModel,1);

for i=1:nGauss
    mu0(:,i) = [2,2]'; % + normrnd(i,1,[nState,1]);
    cov0(:,:,i) = 5*eye(nState); % + normrnd(0,i/2)*eye(nState);
end
sig0 = cov0(1,1,:);

muFinal = mu0;

% Initializations
x = x0;
mu = mu0;
sig = sig0;
cov = cov0;

traj = [mu];
traj_true = [x];


while(max(abs(muFinal - goal)) > 0.1)
    
    [u_plan,mu_plan,s_plan] = createPlan(mu,cov);
    for t=1:T-1
        for i=1:nGauss
            cov(:,:,i) = s_plan(:,:,t)*eye(nState);
        end
        
        % Choosing maximum liklihood model for LQR Control
        wts = model_wts(mu_plan(:,:,t),cov);
        [val,idx] = max(wts);
        
        % LQR_Control
        %         u_local = blqr_fn(mu,sig,u_plan(:,t),mu_plan(:,:,t),mA(:,:,idx),mB(:,:,idx),mC(:,:,idx),T-t);
        u_local = blqr_fn(mu,sig,u_plan(:,t+1),mu_plan(:,:,t),s_plan(:,:,t),mA(:,:,idx),mB(:,:,idx),mC(:,:,idx),1);
        xNew = simulator(x,mu,u_local);
        
        % EKF function for actual dynamics
        wts2 = model_wts(mu,cov);
        [mu,sig] = EKFupdate(xNew,mu,sig,u_local,mA,mB,mC,wts2);
        
        x = xNew;
        
        if (max(abs(mu-mu_plan(:,:,t+1)))> theta)
            disp('breaking loop')
            break;
        end 
                
        traj = [traj,mu];
        traj_true = [traj_true, x];
    end
    
    muFinal = mu;
end

% xx = medfilt1(traj(1,:),10);
% yy = medfilt1(traj(2,:),10);
% % yy = spline(traj(1,:),traj(2,:),xx);
% % yy = fit(traj(1,:)',traj(2,:)','cubicinterp');
% % yy = csaps(traj(1,:),traj(2,:),0.5,xx);
% xx = [traj(1,1),xx(2:end),traj(1,end)];
% yy = [traj(2,1),yy(2:end),traj(2,end)];
% traj2 = [xx;yy];

figure(1);
hold on
plot(traj(1,:),traj(2,:),'r');
plot(traj_true(1,:),traj_true(2,:),'b');

% plot(traj_wo_err(1,:),traj_wo_err(2,:),'b--');
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');

plot(traj_true(1,1),traj_true(2,1),'mo');
plot(traj_true(1,end),traj_true(2,end),'bx');
hold off

% figure(2)
% hold on
% % plot(yy,traj(1,:),traj(2,:),'b');
% plot(xx,yy,'r');
%
% plot(traj(1,1),traj(2,1),'mo');
% plot(traj(1,end),traj(2,end),'bx');
% hold off
save('test1.mat', 'traj');
save('test1_true_traj.mat', 'traj_true');








