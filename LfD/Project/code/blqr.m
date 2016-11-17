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

t_f = 10;
t_res = 1;

x = [2.5,0]';
m = [2,2]';
sig = 5*eye(nState);
% s = sig(1,1);
u = zeros(2,1);

traj = [m];
traj_wo_err = [m];
U = [u];

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.1*eye(nState);
% Q = zeros(nState);
R = 0.5*eye(nInput);
Q_f = 10*eye(nState);%20*eye(nState);
u_max = 0.1;
labda = 200;


% Extende Matrices
A_ext = [[A'; 1e-16*ones(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 10]' ;

%% Rollout MAtrices
A_ext_dummy = A_ext;
K = zeros(nInput,nState+1,t_f/t_res);
F = [[Q_f'; zeros(1,nState)]';0 0 labda]';  %Terminal Cost
%% LQR Control
count = 1;
for t=1:t_res:t_f  
    W = 0.5*(5.0-m(1))^2*eye(nOutput);
    gamma = A*sig*A';
    dsig = sig*(A+A') - gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A'))) + ...
        gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A')*C'))*((C*gamma*C' + W)\(C*gamma))...
        - sig*(A+A')*C'*((C*gamma*C' + W)\(C*gamma));
    
    s = dsig(1,1);
    d_s_m = gamma*C'*((C*gamma*C' + W)\((m(1)-5)*eye(size(W)))*(C*gamma*C' + W)\(C*gamma));
%     A_ext(3,:) = [d_s_m(1,1) 0 s];
    A_ext(3,3) = s;
       
    [K,S3] = finiteLQR(t_f/t_res,A_ext,B_ext,Q_ext,R,F,t_res);
    u = -K(:,:,count)*[m;s];
 
    m = A*m + B*u + normrnd(0,sig(1,1),[nState,1]);   
    sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));  
    
    % Data saving
    traj = [traj, m];
    U = [U, u];
    count = count+1;
end


%% Plot smooth trajectory
nPts = 10;
xNew = [traj(1,1)];
yNew = [traj(2,1)];
for i=1:size(traj,2)-1
    dummy = linspace(traj(1,i),traj(1,i+1),nPts);
    xNew = [xNew,dummy(2:end)];
    dummy = linspace(traj(2,i),traj(2,i+1),nPts);
    yNew = [yNew,dummy(2:end)];
end

% yy = spline(traj(1,:),traj(2,:),xNew);
% polyfit_y = polyfit(traj(1,:),traj(2,:),4);
% yy = polyval(polyfit_y,xNew);

figure(1);
hold on
plot(traj(1,:),traj(2,:),'r');

% plot(traj_wo_err(1,:),traj_wo_err(2,:),'b--');
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
hold off

% figure(2)
% plot(xNew,yy,'b');
save('traj_data5.mat', 'traj');

% %% Visualize the value function
% figure(2)
% X = linspace(min(X)-1,max(X)+1);
%
% %% Value Function
% J = X'*Q*X + U'*R*U;
% pcolor(X(1,:),X(2,:),J);
% colormap(gray);
% shading flat
% shading interp

    %     %% Rollout for Final Value
    %     m_dummy = m;
    %     s_dummy = s;
    %     sig_dummy = sig;
    %     count2 = 1;
    %     for j=t:t_res:t_f
    %         W_dummy = 0.5*(5.0-m_dummy(1))^2*eye(nOutput);
    %         A_ext_dummy(3,3) = s_dummy;
    %         u_dummy = -K(:,:,count2)*[m_dummy;s_dummy];
    %         m_dummy = A*m_dummy + B*u_dummy;
    %         gamma = A*sig_dummy*A';
    %         dsig_dummy = sig_dummy*(A+A') - gamma*C'*((C*gamma*C' + W_dummy)\(C*sig_dummy*(A+A'))) + ...
    %             gamma*C'*((C*gamma*C' + W_dummy)\(C*sig_dummy*(A+A')*C'))*((C*gamma*C' + W_dummy)\(C*gamma))...
    %             - sig_dummy*(A+A')*C'*((C*gamma*C' + W_dummy)\(C*gamma));
    %
    %         s_dummy = dsig_dummy(1,1);
    %         sig_dummy = gamma - gamma*C'*((C*gamma*C' + W_dummy)\(C*gamma));
    %         count2 = count2 + 1 ;
    %     end
    %
    %% Actual update
    %     F = [[Q_f'; zeros(1,nState)]';0 0 labda]'*[m_dummy;s_dummy];  %Terminal Cost

    
    
    %% Controlling Maximum control 
        
    %     for j=1:nInput
    %         if abs(u(j)) > u_max
    %             u(j) = (u(j)/abs(u(j)))*u_max;
    %         end
    %     end
