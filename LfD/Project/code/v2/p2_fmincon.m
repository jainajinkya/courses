% Direct Transcription
clearvars
close all
clc

nState = 2;
nInput = 2;
nOutput = 2;

T = 50;
delta = 1;
nSegments = round(T/delta);


%Initial Conditions
m0 = [2,2]';
sig0 = 5*eye(nState);
s0 = sig0(1,1);
u0 = zeros(nInput,1);
goal = [0,0]';

%System Dynamics
A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];

%Objective function
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
labda = 200;

%% Optimization
%Optimization routine
x0 = rand((nState+1+nInput)*nSegments,1);
x0(1:nState,1) = m0;
x0(nState*nSegments+1,1) = s0;
x0((nState+1)*nSegments+1:(nState+1)*nSegments+nInput,1) = u0;

opti_A = [];
opti_B = [];
opti_Aeq = [];
opti_Beq = [];
lb = -10*ones(size(x0,1),1);
ub = 10*ones(size(x0,1),1);
nonlcon = @(x)covCons(x,x0,nState,nInput,nSegments,delta);

% options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',50000);
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[xfinal,fval,exitflag] = fmincon(@(x)obj_fn(x,nState,nSegments,Q,R,labda,goal),x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,nonlcon,options);


%%%%%%%%%%%Visualization%%%%%%%%%
traj = [];
for i=1:2:nSegments*nState
   traj = [traj, [xfinal(i,1);xfinal(i+1,1)]]; 
end

% figure(1);clf;
% hold on
% plot(traj(1,1:end),traj(2,1:end),'r--','LineWidth',2);
% % scatter(traj(1,:),traj(2,:));
% plot(traj(1,1),traj(2,1),'mo', 'markers',12);
% plot(traj(1,end),traj(2,end),'gx','markers',12);
% hold off
% 
% traj

xx = medfilt1(traj(1,:),10);
yy = medfilt1(traj(2,:),10);
% yy = spline(traj(1,:),traj(2,:),xx);
% yy = fit(traj(1,:)',traj(2,:)','cubicinterp');
% yy = csaps(traj(1,:),traj(2,:),0.5,xx);
xx = [traj(1,1),xx(1:end),traj(1,end)];
yy = [traj(2,1),yy(1:end),traj(2,end)];
traj2 = [xx;yy];

figure(1);
hold on
plot(traj(1,:),traj(2,:),'r');

% plot(traj_wo_err(1,:),traj_wo_err(2,:),'b--');
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
hold off

figure(2)
hold on
% plot(yy,traj(1,:),traj(2,:),'b');
plot(xx,yy,'r');

plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
hold off
save('traj_data_opt1.mat', 'traj2');

