function [wtsNew] = model_wts(mu,cov)
chpts = [[0,0,0,0,0,0]',[0,0,0,0,0,6*pi-0.1]'];  % Number of chnagepoints should be dependent on nModels

gmm = gmdistribution(mu',cov);
wtsNew = cdf(gmm,chpts');
wtsNew = [wtsNew(1,1);diff(wtsNew);1-wtsNew(end,1)];
wtsNew = wtsNew/norm(wtsNew,1);
wtsNew = flipud(wtsNew);
% wtsNew = 1/3*ones(3,1);
end

