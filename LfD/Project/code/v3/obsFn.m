function z = obsFn(mu,W)
global nState
z = mu + mvnrnd(zeros(nState,1),W)';
end