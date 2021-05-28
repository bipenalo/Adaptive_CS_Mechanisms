% Fig. 3 revisited
close all;
%% contrast normalization stage 92%
% large stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm92_lg.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm92_lg_time.mat')
% small stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm92_sm.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm92_sm_time.mat')
% contrast normalization stage 5.5%
% large stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm55_lg.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm55_lg_time.mat')
% small stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm55_sm.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/norm55_sm_time.mat')

% in space
fig = figure(1);
subplot(1,2,1)
title('Small stimulus');
space= linspace(1, 1500, 1500);
hold on 
s = [1001, 2500];
h1 = plot(space, norm_55_sm(s(1):s(2), 25),'--k', 'LineWidth', 2.5);
h2 = plot(space, norm_92_sm(s(1):s(2), 55), 'Color', [0.5 0.5 0.5], 'LineWidth', 2.5);
legend([h1,h2],'2.8%','92%')
hold off
subplot(1,2,2)
title('Large stimulus');
hold on 
h1 = plot(space, norm_55_lg(s(1):s(2), 25),'--k', 'LineWidth', 2.5);
h2 = plot(space, norm_92_lg(s(1):s(2), 55),'Color', [0.5 0.5 0.5], 'LineWidth', 2.5);
legend([h1,h2],'2.8%','92%')
hold off
% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Membrane potential', 'FontSize', 16, 'FontWeight', 'bold');
xlabel(han,'Space', 'FontSize', 16, 'FontWeight', 'bold');


%%
% Adaptive C-S normalization stage 92%
% large stimulus
% close all;
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/lg_92.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/lg_92_time.mat')
% small stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/sm_92.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/sm_92_time.mat')
% contrast normalization stage 5.5%
% large stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/lg_55.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/lg_55_time.mat')
% small stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/sm_55.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/sm_55_time.mat')

% contrast normalization stage 2.8%
% small stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/sm_28.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/sm_28_time.mat')
% small stimulus
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/lg_28.mat')
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/lg_28_time.mat')


lg_92(lg_92 < 0) = 0.0;
sm_92(sm_92 < 0) = 0.0;
sm_55(sm_55 < 0) = 0.0;
lg_55(lg_55 < 0) = 0.0;
sm_28(sm_28 < 0) = 0.0;
lg_28(lg_28 < 0) = 0.0;



% in space
figure(2);
space= linspace(1, 1500, 1500);
s = [1001, 2500];
hold on 
xlabel('Space');
ylabel('Membrane voltage');
plot(space, sm_28(s(1):s(2), 97), '--k', 'LineWidth', 2.75);
% plot(space, sm_55(300:3300, 25), '--');
plot(space, sm_92(s(1):s(2), 30),  'Color', [0.5 0.5 0.5], 'LineWidth', 2.5);
% plot(space, lg_55(300:3300, 100), '--', 'LineWidth', 2.75);
plot(space,1.2* lg_28(s(1):s(2), 100), '--k', 'LineWidth', 2.5);
plot(space, lg_92(s(1):s(2), 45), 'Color', [0.5 0.5 0.5],'LineWidth', 2.75);

legend('2.8%', '92%');
hold off

figure(3)
plot(lg_28_times_, lg_28(1800, :));
