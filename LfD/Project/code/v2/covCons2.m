function [c,ceq] = covCons2(x,x0,mA,mB,mC,wts,nGauss,nInput,nSegments,delta)
nState = size(mA,1);

% m = x(1:nState*nGauss*nSegments,:);
% s = x(nState*nGauss*nSegments+1:(nState+1)*nGauss*nSegments,:);
u = x((nState+1)*nGauss*nSegments+1:end);

% m = reshape(m,[nState,nGauss,nSegments]);
% s = reshape(s,[1,nGauss,nSegments]);
u = reshape(u,[nInput,nSegments]);

% Initialization of the new vectors
m_new = rand(nState,nGauss,nSegments);
s_new = rand(1,nGauss,nSegments);
u_new = u;

m_new(:,:,1) = reshape(x0(1:nState*nGauss,1),[nState,nGauss]);
s_new(:,:,1) = reshape(x0(nState*nGauss*nSegments+1:nState*nGauss*nSegments+nGauss,1),[1,nGauss]);

mu = m_new(:,:,1);
for i=1:nGauss
    cov(:,:,i) = s_new(:,i,1)*eye(nState);
end

for i=1:nSegments-1    
    for t=1:delta
        [mu,cov] = modelDynamics(x,mu,cov,u(:,i),mA,mB,mC,wts);
        [wts] = model_wts(mu,cov);
    end
    m_new(:,:,i+1) = mu;
    s_new(:,i+1) = cov(1,1);
end

%% Converting matrices back to the vector format
m_new = reshape(m_new,[nState*nGauss*nSegments,1]);
s_new = reshape(s_new,[nSegments*nGauss,1]);
u_new = reshape(u_new,[nInput*nSegments,1]);

x_new = [m_new;s_new;u_new];

% ceq = x_new - x;
% c = [];

delta = 10.0;
ceq = [];
c = [x_new - x - delta*ones(size(x0,1),1); x - x_new - delta*ones(size(x0,1),1)];

end