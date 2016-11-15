%% a generic linear control problem:
clear
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

m1 = rand(nM,1);
s1 = 5;
sig = s1*eye(nM);
W = 0.5*(5.0-m1(1,1))^2*eye(nM);

T = 20;

for i=1:T
    
    cvx_begin
    variables m(nM,T) u(nU,T) s(1,T) J(1,T);
    minimize( sum(J) )
    subject to
    for t=1:T-1
        m(:,t+1) == A*m(:,t) + B*u(:,t) + normrnd(0,sig(1,1),[2,1]);
        
        %     W = 0.5*(5.0-m(1,t+1))^2*eye(nM);
%         W = 0.1*eye(nM);
        gamma = A*sig*A';
        dsig = sig*(A+A') - gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A'))) + ...
            gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A')*C'))*((C*gamma*C' + W)\(C*gamma))...
            - sig*(A+A')*C'*((C*gamma*C' + W)\(C*gamma));
        
        s(:,t+1) == dsig(1,1);
        sig = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
    end
    for t=1:T-1
        J(1,t) >= m(:,t)'*Q*m(:,t) + u(:,t)'*R*u(:,t);
    end
    J(1,T) >= m(:,T)'*Q*m(:,T) + u(:,T)'*R*u(:,T) + s(:,T)'*labda*s(:,T);
    
    m(:,1) == m1;
    s(:,1) == s1;
    cvx_end
    
    M(:,i) = m(:,i);
    U(:,i) = u(:,i);
    W = (0.5*(5.0-M(1,i))^2)*eye(nM);
    i
end

figure; plot(M');

figure(1);clf;
hold on
plot(M(1,1:end-1),M(2,1:end-1),'r--','LineWidth',2);
% scatter(traj(1,:),traj(2,:));
plot(M(1,1),M(2,1),'mo', 'markers',12);
plot(M(1,end-1),M(2,end-1),'gx','markers',12);
hold off
save('traj_cvx1.mat', 'M');

% % let's execute control policy:
% 
% m_run(:,1) = m1;
% 
% for t=1:T-1
%     m_run(:,t+1) = A*m_run(:,t) + B*u(:,t);
% end
% 
% figure; plot(m_run');