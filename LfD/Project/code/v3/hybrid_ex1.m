clearvars
close all
clc

global nState nInput nOutput nModel nGauss T goal Q R Q_f labda mA mB mC
nState = 6;
nInput = 6;
nOutput = 6;
nModel = 3; % Check number of changepoints
nGauss = 1;
goal = [0,0,0,0,0,6*pi]';
T = 12;

% Dynamics Matrics
[mA,mB,mC] = dynamicsTemplate();

% Cost Matrices
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
Q_f = 20*eye(nState);
labda = 2000;

theta = 1.0;

% GMM Intialize
mu = zeros(nState,nGauss);
cov = zeros(nState,nState,nGauss);
mProb = zeros(nModel,1);

for i=1:nGauss
   mu(:,i) = [2,2,2,pi,pi/2,pi]'; % + normrnd(i,1,[nState,1]);  
   cov(:,:,i) = 5*eye(nState); % + normrnd(0,i/2)*eye(nState);
end
sig = cov(1,1,:);

muFinal = mu;

while(max(abs(muFinal - goal)) > 0.1)
    [u_plan,mu_plan,s_plan] = createPlan(mu,cov);
    traj = [mu];

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

        % EKF function for actual dynamics
        wts2 = model_wts(mu,cov);
        [mu,sig] = EKFupdate(mu,mu,sig,u_local,mA,mB,mC,wts2);

        traj = [traj,mu];
        
        if (max(abs(mu-mu_plan(:,:,t+1)))> theta)
            disp('breaking loop')
            break;
        end    
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

% plot(traj_wo_err(1,:),traj_wo_err(2,:),'b--');
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
hold off

% figure(2)
% hold on
% % plot(yy,traj(1,:),traj(2,:),'b');
% plot(xx,yy,'r');
% 
% plot(traj(1,1),traj(2,1),'mo');
% plot(traj(1,end),traj(2,end),'bx');
% hold off
save('test2.mat', 'traj');








