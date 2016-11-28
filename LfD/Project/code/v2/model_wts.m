function [wtsNew] = model_wts(mu,cov)
chpts = [[1,0]',[2,0]'];

gmm = gmdistribution(mu',cov);
wtsNew = cdf(gmm,chpts');
wtsNew = [wtsNew(1,1);diff(wtsNew);1-wtsNew(end,1)];
wtsNew = wtsNew/norm(wtsNew,1);
end
