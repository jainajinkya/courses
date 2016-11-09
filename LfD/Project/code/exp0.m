
% %% LQG Control Exp-1
% nState = 2;
% nInput = 2;
% nOutput = 2;
% t_f = 100;
% 
% A = [1 0; 0 1];
% B = [1 0; 0 1];
% C = [1 0; 0 1];
% V = 0.02*eye(nState);
% W = 0.05*eye(nOutput);
% mu = zeros(nState);
% 
% x = [2.5; 0.0];
% y = [0.0; 0.0];
% x_hat = [2.0; 2.0];
% y = C*x + normrnd(mu,W);
% 
% 
% traj2 = [x];
% 
% Q = 0.5*eye(nState);
% R = 0.5*eye(nInput);
% F = 2*eye(nState);
% 
% K = zeros(nOutput,nOutput,t_f);
% L = zeros(nState,nState,t_f);
% S = zeros(nState,nState,t_f);
% P = zeros(nState,nState,t_f);
% S(:,:,t_f)= F;
% P(:,:,1)= zeros(nState);
% 
% for t=t_f-1:-1:1
%     S(:,:,t) = Q + A'*(S(:,:,t+1) - S(:,:,t+1)*B*((B'*S(:,:,t+1)*B + R)\(B'*S(:,:,t+1))))*A;
%     L(:,:,t) = (B'*S(:,:,t+1)*B + R)\(B'*S(:,:,t+1)*A);
% end
% 
% for t=1:t_f
% %    W = 0.5*(5-x(1))^2*eye(nOutput);
%    u = -L(:,:,t)*x_hat;
%    x = A*x + B*u + normrnd(mu(1,1),V(1,1),[nState,1]);
%    y = C*x + normrnd(mu(1,1),W(1,1),[nOutput,1]);
%   
%    if t==t_f
%       continue;
%    else
%        P(:,:,t+1) =  A*(P(:,:,t) - P(:,:,t)*C'*((C*P(:,:,t)*C' + W)\C*P(:,:,t)))*A' + V;
%        K(:,:,t+1) = P(:,:,t)*C'/(C*P(:,:,t)*C' + W);
%        x_hat = A*x_hat + B*u + K(:,:,t+1)*(y - C*(A*x_hat + B*u));
%    end
%   1  
%    traj2 = [traj2,x];
% end
% 
% figure(2);clf;
% scatter(traj2(1,:),traj2(2,:),'r');
% hold on
% plot(traj2(1,t_f),traj2(2,t_f),'bx');
% hold off
