function [wtsNew] = model_wts(gmDist,nModel)
chpts = [[1,0]',[2,0]',[3,0]'];

% wtsNew = zeros(nModel,1);
% for i=1:size(chpts,1)
%     wtsNew(i) = cdf(gmDist,chpts(i));
% end
wtsNew = cdf(gmDist,chpts);

end

