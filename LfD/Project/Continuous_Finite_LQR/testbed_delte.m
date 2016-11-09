a=[1.1 5.2 9.25 10];
n_pts = 4; %specify the number of intervening points
a_desired = cumsum([a;repmat([diff(a)/n_pts,0],n_pts-1,1)]);
a_desired = a_desired(1:end-n_pts+1);