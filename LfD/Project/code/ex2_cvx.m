%% a generic linear control problem:
clear all
close
clc

nM = 2;
nU = 2;

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];

Q = 0.5*eye(nM);
R = 0.5*eye(nU);
labda = 1000;

m1 = [2,2]';
s1 = 5;
sig = s1*eye(nM);
gamma = A*sig*A';
m(:,1) = m1;


T = 20;
k = 20;
delta = T/k;

m_dummy = m1;

cvx_begin
variables m(nM,k) u(nU,k) s(1,k) J(1,k) sig(2,2) 
expressions C(2,2) gamma(2,2)
minimize( sum(J) )
subject to
C = [1 0; 0 1];
gamma = A*sig*A';

for i=1:k-1
%     W = (0.5*(5.0-m(1,1))^2)*eye(nM);
    det_inv(C);
    for t=1:delta
        m_dummy = m_dummy + A*m_dummy + B*u(:,k) + normrnd(0,0.1,[2,1]);
        sig = gamma - gamma*C'*(det_inv(C*gamma*C' + W)*eye(nM))*(C*gamma);
        W = (0.5*(5.0-m_dummy(1,1))^2)*eye(nM);
    end
    m(:,i+1) == m_dummy;
    s(:,i+1) == sig(1,1);
end
for i=1:k-1
    J(1,i) >= m(:,i)'*Q*m(:,i) + u(:,i)'*R*u(:,i);
end
J(1,k) >= m(:,k)'*Q*m(:,k) + u(:,k)'*R*u(:,k) + s(:,k)'*labda*s(:,k);

m(:,1) == m1;
s(:,1) == s1;
cvx_end


figure(1);clf;
hold on
plot(m(1,:),m(2,:),'r--','LineWidth',2);
% scatter(traj(1,:),traj(2,:));
plot(m(1,1),m(2,1),'mo', 'markers',12);
plot(m(1,end),m(2,end),'gx','markers',12);
hold off


% 
% M(:,i) = m(:,i);
% U(:,i) = u(:,i);
% W = (0.5*(5.0-m(1,i))^2)*eye(nM)


% for i=1:T
%     cvx_begin
%     variables m(nM,T) u(nU,T) s(1,T) J(1,T)
%     minimize( sum(J) )
%     subject to
%     for t=1:T-1
%         m(:,t+1) == A*m(:,t) + B*u(:,t) + normrnd(0,0.1,[2,1]);
%         sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
%         s(:,t+1) == sig(1,1);
%     end
%     for t=1:T-1
%         J(1,t) >= m(:,t)'*Q*m(:,t) + u(:,t)'*R*u(:,t);
%     end
%     J(1,T-2:T) >= m(:,T)'*Q*m(:,T) + u(:,T)'*R*u(:,T) + s(:,T)'*labda*s(:,T);
%
%     m(:,1) == m1;
%     s(:,1) == s1;
%     cvx_end
%
%     M(:,i) = m(:,i);
%     U(:,i) = u(:,i);
%     W = (0.5*(5.0-m(1,i))^2)*eye(nM)
% end
% 
% figure(1);clf;
% hold on
% plot(M(1,1:end-1),M(2,1:end-1),'r--','LineWidth',2);
% % scatter(traj(1,:),traj(2,:));
% plot(M(1,1),M(2,1),'mo', 'markers',12);
% plot(M(1,end-1),M(2,end-1),'gx','markers',12);
% hold off
% save('traj_cvx1.mat', 'M');

% % let's execute control policy:
%
% m_run(:,1) = m1;
%
% for t=1:T-1
%     m_run(:,t+1) = A*m_run(:,t) + B*u(:,t);
% end
%
% figure; plot(m_run');