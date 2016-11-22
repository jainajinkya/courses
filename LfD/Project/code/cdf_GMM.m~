function [wtsNew] = cdf_GMM(mu,cov,wtsGMM, wts)
nDim = size(mu,1);
nGauss = size(mu,2);

chpts = [[1,0]',[2,0]',[3,0]'];

for cp = 1:size(chpts,1)
    for i=1:nDim
        for k=1:nGauss
            dummy(i) = wtsGMM(k)*normcdf(chpts(i,cp),mu(i,k),cov(i,i,k));
        end
    end
end



normcdf(chpts(:,i),mu())


end

