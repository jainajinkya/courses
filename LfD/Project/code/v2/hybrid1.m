% clear
close all
clc

nState = 2;
nInput = 2;
nOutput = 2;
nModel = 3;
nGauss = 5;

% x = [2.5,0]';
% u = [0,0]';
goal = [0,0]';

% GMM Intialize
mu = zeros(nState,nGauss);
cov = zeros(nState,nState,nGauss);
mProb = zeros(nModel,1);
wts = (1/nModel)*ones(nModel,1);

for i=1:nGauss
   mu(:,i) = [2,2]' + normrnd(i,1,[nState,1]);  
   cov(:,:,i) = 5*eye(nState) + normrnd(0,i/2)*eye(nState);
end
u0 = zeros(nInput,1);

T = 10;
delta = 1;
nSegments = round(T/delta);

% traj = [m];
% traj_wo_err = [m];
% U = [u];

%% Model Dynamics
mA = zeros(nState,nState,nModel);
mB = zeros(nState,nState,nModel);
mC = zeros(nState,nState,nModel);

for i=1:nModel
   mA(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
   mB(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
   mC(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
end

% Cost Matrices
Q = 0.01*eye(nState);
R = 0.5*eye(nInput);
Q_f = 20*eye(nState);
labda = 2000;

%% Optimization
%Optimization routine
% x0 = rand(((nState+1)*nGauss +nInput)*nSegments,1);
% x0(1:nState*nGauss,1) = reshape(mu,[nState*nGauss,1]);
% x0(nState*nGauss*nSegments+1:nState*nGauss*nSegments+nGauss,1) = cov(1,1,:);
% x0((nState+1)*nGauss*nSegments+1:(nState+1)*nGauss*nSegments+nInput,1) = u0;
% 
opti_A = [];
opti_B = [];
opti_Aeq = [];
opti_Beq = [];
lb = -20*ones(size(x0,1),1);
ub = 20*ones(size(x0,1),1);
constraintRelax = 10.0;
nonlcon = @(x)covCons2(x,x0,mA,mB,mC,wts,nGauss,nInput,nSegments,delta,constraintRelax/4);

options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',1000000);
% options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[xfinal,fval,exitflag] = fmincon(@(x)obj_fn(x,nState,nSegments,Q,R,labda,goal),x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,nonlcon,options)
% 
% [xfinal,fval,exitflag] = fmincon(@(x)0,x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,nonlcon,options)





% %%%%%%%%%%%Visualization%%%%%%%%%
% traj = [];
% for i=1:2:nSegments*nState
%    traj = [traj, [xfinal(i,1);xfinal(i+1,1)]]; 
% end
% 
% % figure(1);clf;
% % hold on
% % plot(traj(1,1:end),traj(2,1:end),'r--','LineWidth',2);
% % % scatter(traj(1,:),traj(2,:));
% % plot(traj(1,1),traj(2,1),'mo', 'markers',12);
% % plot(traj(1,end),traj(2,end),'gx','markers',12);
% % hold off
% % 
% % traj
% 
% xx = medfilt1(traj(1,:),10);
% yy = medfilt1(traj(2,:),10);
% % yy = spline(traj(1,:),traj(2,:),xx);
% % yy = fit(traj(1,:)',traj(2,:)','cubicinterp');
% % yy = csaps(traj(1,:),traj(2,:),0.5,xx);
% xx = [traj(1,1),xx(1:end),traj(1,end)];
% yy = [traj(2,1),yy(1:end),traj(2,end)];
% traj2 = [xx;yy];
% 
% figure(1);
% hold on
% plot(traj(1,:),traj(2,:),'r');
% 
% % plot(traj_wo_err(1,:),traj_wo_err(2,:),'b--');
% plot(traj(1,1),traj(2,1),'mo');
% plot(traj(1,end),traj(2,end),'bx');
% hold off
% 
% figure(2)
% hold on
% % plot(yy,traj(1,:),traj(2,:),'b');
% plot(xx,yy,'r');
% 
% plot(traj(1,1),traj(2,1),'mo');
% plot(traj(1,end),traj(2,end),'bx');
% hold off
% save('traj_data_opt1.mat', 'traj2');




% nModel = 3;
% nGauss = 5;
% 
% mu = zeros(nState,nGauss);
% cov = zeros(nState,nState,nGauss);
% mProb = zeros(nModel,1);
% 
% mA = zeros(nState,nState,nModel);
% mB = zeros(nState,nState,nModel);
% mA_ext = zeros(nState+1,nState+1,nModel);
% mB_ext = zeros(nState+1,nState+1,nModel);
% 
% for i=1:nGauss
%    mu(:,i) = [2,2]' + normrnd(0,0.1,[nState,1]);  
%    cov(:,:,i) = sig + normrnd(0,0.1)*eye(nState);
% end
% 
% for i=1:nModel
%    mA(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
%    mB(:,:,i) = eye(nState) + normrnd(0,0.1)*eye(nState);
%    mA_ext = [[mA(:,:,i)'; zeros(1,nState)]';0 0 0]';
%    mB_ext = [mB(:,:,i); 0 0];
% end
