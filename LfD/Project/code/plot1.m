clear
close
clc

traj = importdata('traj_cvx1.mat');
X = traj(1,:);
Y = traj(2,:);

figure(1);clf;
X_plot = linspace(min(X)-1,max(X)+1);
Y_plot = linspace(min(Y)-1,max(Y)+1);
C = ones(length(Y_plot),length(X_plot));

for i=1:size(C,1)
    for j=1:size(C,2)
        C(i,j) = -0.5*(5-X_plot(j))^2;
    end
end
pcolor(X_plot,Y_plot,C);
colormap(gray);
shading flat
shading interp
% axis([-1,7,-2,5])
hold on

% %% Normal Plot
% figure(3);
% hold on

plot(traj(1,:),traj(2,:),'r--','LineWidth',2);
% scatter(X,Y);
plot(traj(1,1),traj(2,1),'mo', 'markers',12);

plot(traj(1,end),traj(2,end),'gx','markers',12);

title('Trajectory of the Belief Mean')
xlabel('x')
ylabel('y')

hold off

%%

% 
%% Smooth Trajectory
% figure(4)
% % polyfit_y = polyfit(X,Y,8);
% % fit_y = polyval(polyfit_y,X);
% xx = smooth(X,0.1,'loess');
% yy = smooth(Y,0.1,'loess');
% plot(xx,yy,'r-','LineWidth',2);
% hold on
% % plot(xx,yy,'r');
% scatter(xx,yy);
% plot(traj(1,1),traj(2,1),'mo');
% plot(traj(1,end),traj(2,end),'bx');
% hold off




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Experimentation %%%%%%%%%%%%%%%%%
% figure(1);clf;
% Im = zeros(length(Y),length(X));
% 
% for i=1:size(Im,1)
%     for j=1:size(Im,2)
%         Im(i,j) = 0.5*(5-X(j)) - 3;
%     end
% end
% 
% X_scaled = X*(size(Im,2)/(max(X)-min(X)));
% Y_scaled = Y*(size(Im,1)/(max(Y)-min(Y)));
% 
% figure(1)
% imshow(Im)
% % hold on
% 
% %Making X and Y of the same length
% if (length(X_scaled) > length(Y_scaled))
%    Y_scaled = interp1(linspace(1,length(Y_scaled),length(Y_scaled)),Y_scaled,X_scaled,'pchip');
%     
% elseif (length(Y_scaled) > length(X_scaled))
%     X_scaled = interp1(linspace(1,length(X_scaled),length(X_scaled)),X_scaled,Y_scaled,'pchip');
% end
% 
% % plot(traj(1,:),traj(2,:),'r');
% % plot(traj(1,t_f+1),traj(2,t_f+1),'bx');
% figure(2);
% plot(X_scaled,Y_scaled,'r');
% hold on
% plot(X_scaled(1),Y_scaled(1),'mo');
% plot(X_scaled(t_f+1),Y_scaled(t_f+1),'bx');
% hold off