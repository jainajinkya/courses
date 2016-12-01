function z = obsFn(x,mu,W)
global nState
z = mu + 0.1*rand(nState,1);
end