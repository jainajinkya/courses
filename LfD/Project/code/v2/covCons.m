function [c,ceq] = covCons(x,x0,nState,nInputs,nSegments,delta)
m = x(1:nState*nSegments,:);
s = x(nState*nSegments+1:(nState+1)*nSegments,:);
u = x((nState+1)*nSegments+1:end);

m = reshape(m,[nState,nSegments]);
s = reshape(s,[1,nSegments]);
u = reshape(u,[nInputs,nSegments]);

%System Dynamics
A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];

% Initialization of the new vectors
m_new = rand(nState,nSegments);
s_new = rand(1,nSegments);
u_new = u;

m_new(:,1) = x0(1:nState,1);
s_new(:,1) = x0(nState*nSegments+1,1);
sig = s_new(1,1)*eye(nState);

m_dummy = m_new(:,1);

for i=1:nSegments-1    
    for t=1:delta
        W = (0.5*(5.0-m_dummy(1,1))^2)*eye(nState);
        gamma = A*sig*A';
        m_dummy = A*m_dummy + B*u(:,i); % + normrnd(0,0.1,[2,1]);
        sig = gamma - gamma*C'*(inv(C*gamma*C' + W))*(C*gamma);
    end
    m_new(:,i+1) = m_dummy;
    s_new(:,i+1) = sig(1,1);
end

% %% Old Version
% m_new(:,1) = x0(1:nState,1);
% s_new(:,1) = x0(nState*nSegments+1,1);
% sig = s_new(1,1)*eye(nState);
% 
% for i=1:nSegments-1
%     W = (0.5*(5.0-m(1,i))^2)*eye(nState);
%     gamma = A*sig*A';
%     sig = gamma - gamma*C'*(inv(C*gamma*C' + W))*(C*gamma);
% 
%     m_new(:,i+1) = A*m(:,i) + B*u(:,i);
%     s_new(:,i+1) = sig(1,1);
% end


%% Converting matrices back to the vector format
m_new = reshape(m_new,[nState*nSegments,1]);
s_new = reshape(s_new,[nSegments,1]);
u_new = reshape(u_new,[nInputs*nSegments,1]);

x_new = [m_new;s_new;u_new];

ceq = x_new - x;
c = [];

end