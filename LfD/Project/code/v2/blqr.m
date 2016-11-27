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
t_res = 1;

x = [2.5,0]';
m = [2,2]';
sig = 5*eye(nState);
u = zeros(2,1);
u_max = 0.05;

traj = [m];
traj_wo_err = [m];
U = [u];

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
Q_f = 100*eye(nState);
labda = 2000;


% Extende Matrices
A_ext = [[A'; zeros(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 0]' ;
F = [[Q_f'; zeros(1,nState)]';0 0 labda]';  %Terminal Cost
%% LQR Control
count = 1;
c2 = 0;
for t=1:t_res:t_f-1  
    W = 0.5*(5.0-x(1))^2*eye(nOutput);
    gamma = A*sig*A';
    dg_dsig= A*A';
    ds_dsig = dg_dsig - dg_dsig*C'*(inv(C*gamma*C' + W)*(C*gamma)) ...
              + gamma*C'*(inv(C*gamma*C' + W)*(C*dg_dsig*C')*inv(C*gamma*C' + W)*(C*gamma)) ...
              - gamma*C'*(inv(C*gamma*C' + W)*(C*dg_dsig));

    d_s_m = gamma*C'*(inv(C*gamma*C' + W)*((m(1)-5)*eye(size(W)))*inv(C*gamma*C' + W)*(C*gamma));
    
    A_ext(3,:) = [d_s_m(1,1) 0  ds_dsig(1,1)];

       
    %[K,S3] = finiteLQR(t_f/t_res-t+1,A_ext,B_ext,Q_ext,R,F,t_res);
    
    if(rem(c2,2)==0)
        [K,S3] = finiteLQR(t_f-count+1,A_ext,B_ext,Q_ext,R,F,t_res); 
        c2 = 0;
    end
    
    S3
    u = -K(:,:,c2+1)*[m;sig(1,1)];
    
%     K(:,:,count)
%     [m;sig(1,1)]'*S3(:,:,count+1)*[m;sig(1,1)];
%     u = -K(:,:,1)*[m;sig(1,1)];
%     % Controlling Maximum control
%     
    for j=1:nInput
%         if abs(u(j)) > u_max
            u(j) = (u(j)/abs(u(j)))*u_max;
%         end
    end
        
     
    %% System Dynamics:
    x = A*(x-m) + A*m + B*u;
    m = A*m + B*u + normrnd(0,0.1,[nState,1]);   
    sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
    
    % Data saving
    traj = [traj, m];
    U = [U, u];
    count = count+1;
    c2 = c2+1;
end


% %% Plot smooth trajectory
% nPts = 20;
% xNew = [traj(1,1)];
% yNew = [traj(2,1)];
% for i=1:size(traj,2)-1
%     dummy = linspace(traj(1,i),traj(1,i+1),nPts);
%     xNew = [xNew,dummy(2:end)];
%     dummy = linspace(traj(2,i),traj(2,i+1),nPts);
%     yNew = [yNew,dummy(2:end)];
% end
% yy = spline(traj(1,1:4),traj(2,1:4),traj(1,1:4));
% yy = fit(traj(1,:)',traj(2,:)','cubicinterp');
% yy = csaps(traj(1,:),traj(2,:),0.5,traj(1,:));
% polyfit_y = polyfit(traj(1,:),traj(2,:),4);
% yy = polyval(polyfit_y,xNew);

xx = medfilt1(traj(1,:),10);
yy = medfilt1(traj(2,:),10);
% yy = spline(traj(1,:),traj(2,:),xx);
% yy = fit(traj(1,:)',traj(2,:)','cubicinterp');
% yy = csaps(traj(1,:),traj(2,:),0.5,xx);
xx = [traj(1,1),xx(4:end),traj(1,end)];
yy = [traj(2,1),yy(4:end),traj(2,end)];
traj2 = [xx;yy];

figure(1);
hold on
plot(traj(1,:),traj(2,:),'r');

% plot(traj_wo_err(1,:),traj_wo_err(2,:),'b--');
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
hold off

figure(2)
% plot(yy,traj(1,:),traj(2,:),'b');
plot(xx,yy,'r');
hold on
plot(traj(1,1),traj(2,1),'mo');
plot(traj(1,end),traj(2,end),'bx');
save('traj_data_new1.mat', 'traj2');

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
