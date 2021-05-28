%% Stimulus duration --> 40 ms (Surround Suppressed)
% 5 deg ss (surround suppressed)
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_5deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_5deg_time.mat');
scale = 33;
g = 1.15;
offset = 50;
% Exp. data
dat_5deg_40ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_40ms_.csv');
% obtain average value between 100 ms and 175 ms
row = dat_5deg_40ms(:,1);
indices = (row > 100) & (row < 175);
mean_5deg_40ms = mean(dat_5deg_40ms(indices,2));

%deg5_40ms(deg5_40ms < 0.0) = 0.0;
time1 = scale*deg5_40ms_times_;
logInd = deg5_40ms(1800, :) > 0;
m1 = mean(deg5_40ms(1800, logInd));

% 8 deg ss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_8deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_8deg_time.mat');

% Exp. data
dat_8deg_40ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg8_40ms_.csv');
row = dat_8deg_40ms(:,1);
indices = (row > 100) & (row < 175);
mean_8deg_40ms = mean(dat_8deg_40ms(indices,2));

deg8_40ms(deg8_40ms < 0.0) = 0.0;
time2 = scale*deg8_40ms_times_;
logInd = deg8_40ms(1800, :) > 0;
m2 = mean(deg8_40ms(1800, logInd));


% 11 deg
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_11deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_11deg_time.mat');
% Exp. data
dat_11deg_40ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg11_40ms_.csv');
row = dat_11deg_40ms(:,1);
indices = (row > 100) & (row < 175);
mean_11deg_40ms = mean(dat_11deg_40ms(indices,2));

deg11_40ms(deg11_40ms < 0.0) = 0.0;
time3 = scale*deg11_40ms_times_;
logInd = deg11_40ms(1800, :) > 0;
m3 = mean(deg11_40ms(1800, logInd));

% 14 deg
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_14deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_14deg_time.mat');
% Exp. data
dat_14deg_40ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg14_40ms_.csv');
row = dat_14deg_40ms(:,1);
indices = (row > 100) & (row < 175);
mean_14deg_40ms = mean(dat_14deg_40ms(indices,2));

deg14_40ms(deg14_40ms < 0.0) = 0.0;
time4 = scale*deg14_40ms_times_;
logInd = deg14_40ms(1800, :) > 0;
m4 = mean(deg14_40ms(1800, logInd));

fig = figure(1);
subplot(1, 4, 1)
title({'Stimulus duration 40 ms ------- Surround suppressed'; '5 deg'}, 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
hold on 
ylim([0 50]);
xlim([0 250]);
plot(time1+offset, g*deg5_40ms(1800, :),'k', 'LineWidth', 3);
dat_5deg_40ms_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_40ms_std.csv');
ypos1 = dat_5deg_40ms_std(:,2) - dat_5deg_40ms(:,2);
% yneg1 = -dat_5deg_40ms(51:75,2) + dat_5deg_40ms(1:25,2);

% yposerr= [ypos1, yneg1];
p1 = shadedErrorBar(dat_5deg_40ms(:,1), dat_5deg_40ms(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_5deg_40ms(:,1), dat_5deg_40ms(:,2), '-r', 'LineWidth', 2); 
%p3= createfig_MDtask(dat_5deg_40ms(:,1), dat_5deg_40ms(:,2), 'none');
hold off
subplot(1, 4, 2)
hold on 
title('8 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 50]);
xlim([0 250]);
plot(time2+offset, g*deg8_40ms(1800, :), 'k', 'LineWidth', 3);
dat_8deg_40ms_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg8_40ms_std.csv');
ypos1= dat_8deg_40ms_std(:,2) - dat_8deg_40ms(:,2);
% ypos1 = dat_8deg_40ms(31:60,2) - dat_8deg_40ms(1:30,2);
% yneg1 = -dat_8deg_40ms(61:90,2) + dat_8deg_40ms(1:30,2);
p1 = shadedErrorBar(dat_8deg_40ms(:,1), dat_8deg_40ms(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_8deg_40ms(:,1), dat_8deg_40ms(:,2), '-r', 'LineWidth', 2); 
%p3= createfig_MDtask(dat_5deg_40ms(:,1), dat_5deg_40ms(:,2), 'none');

hold off
subplot(1, 4, 3)
hold on 
title('11 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 50]);
xlim([0 250]);
dat_11deg_40ms_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg11_40ms_std.csv');
ypos1= dat_11deg_40ms_std(:,2) - dat_11deg_40ms(:,2);
p1 = shadedErrorBar(dat_11deg_40ms(:,1), dat_11deg_40ms(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_11deg_40ms(:,1), dat_11deg_40ms(:,2), '-r', 'LineWidth', 2); 
% plot(dat_11deg_40ms(:,1), dat_11deg_40ms(:,2), 'r', 'LineWidth', 2);
plot(time3+offset, g*deg11_40ms(1800, :), 'k', 'LineWidth', 3);
hold off
subplot(1, 4, 4)
hold on 
title('14 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 50]);
xlim([0 250]);
dat_14deg_40ms_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg14_40ms_std.csv');
ypos1= dat_14deg_40ms_std(:,2) - dat_14deg_40ms(:,2);
p1 = shadedErrorBar(dat_14deg_40ms(:,1), dat_14deg_40ms(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_14deg_40ms(:,1), dat_14deg_40ms(:,2), '-r', 'LineWidth', 2); 
% plot(dat_14deg_40ms(:,1), dat_14deg_40ms(:,2), 'LineWidth', 2);
plot(time4+offset, g*deg14_40ms(1800, :), 'k', 'LineWidth', 3);
hold off
% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
han.FontSize = 12;
ylabel(han,'Activity (spikes/s)', 'FontSize', 16, 'FontWeight', 'normal', 'FontName', 'Arial');
xlabel(han,'Time (ms)', 'FontSize', 14, 'FontWeight', 'normal', 'FontName', 'Arial');

% title(han,'yourTitle');

%% Stimulus duration --> 100 ms
% 5 deg ss (surround suppressed)
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_5deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_5deg_time.mat');
% Exp. data
dat_5deg_100ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_100ms_.csv');

deg5_100ms(deg5_100ms < 0.0) = 0.0;
time1 = scale*deg5_100ms_times_;

% 8 deg ss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_8deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_8deg_time.mat');

% Exp. data
dat_8deg_100ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg8_100ms_.csv');

deg8_100ms(deg8_100ms < 0.0) = 0.0;
time2 = scale*deg8_100ms_times_;

% 11 deg ss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_11deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_11deg_time.mat');
% 11 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_11deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_11deg_time_nss.mat');

% Exp. data
dat_11deg_100ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg11_100ms_.csv');

deg11_100ms(deg11_100ms < 0.0) = 0.0;
time3 = scale*deg11_100ms_times_;

% 14 deg
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_14deg.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_14deg_time.mat');
% Exp. data
dat_14deg_100ms = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg14_100ms_.csv');

deg14_100ms(deg14_100ms < 0.0) = 0.0;
time4 = scale*deg14_100ms_times_;

offset = 50;

fig = figure(2);
subplot(1, 4, 1)
title({'Stimulus duration 100 ms ------- Surround suppressed'; '5 deg'}, 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
hold on 
ylim([0 50]);
xlim([0 250]);
% dat_5deg_100ms_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_100ms_std.csv');
ypos1= dat_5deg_100ms(35:68,2) - dat_5deg_100ms(1:34,2);
p1 = shadedErrorBar(dat_5deg_100ms(1:34,1), dat_5deg_100ms(1:34,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_5deg_100ms(1:34,1), dat_5deg_100ms(1:34,2), '-r', 'LineWidth', 2); 
% plot(dat_11deg_40ms(:,1), dat_11deg_40ms(:,2), 'r', 'LineWidth', 2);
plot(time1+offset, g*deg5_100ms(1800, :), 'k', 'LineWidth', 3);hold off
subplot(1, 4, 2)
hold on 
title('8 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ypos1= dat_8deg_100ms(25:48,2) - dat_8deg_100ms(1:24,2);
p1 = shadedErrorBar(dat_8deg_100ms(1:24,1), dat_8deg_100ms(1:24,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_8deg_100ms(1:24,1), dat_8deg_100ms(1:24,2), '-r', 'LineWidth', 2); 
% plot(dat_11deg_40ms(:,1), dat_11deg_40ms(:,2), 'r', 'LineWidth', 2);
plot(time2+offset, g*deg8_100ms(1800, :), 'k', 'LineWidth', 3);
hold off
subplot(1, 4, 3)
hold on 
title('11 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 50]);
xlim([0 250]);
ypos1= dat_11deg_100ms(24:46,2) - dat_11deg_100ms(1:23,2);
p1 = shadedErrorBar(dat_11deg_100ms(1:23,1), dat_11deg_100ms(1:23,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_11deg_100ms(1:23,1), dat_11deg_100ms(1:23,2), '-r', 'LineWidth', 2); 
plot(time3+offset, g*deg11_100ms(1800, :), 'k', 'LineWidth', 3);
hold off
subplot(1, 4, 4)
hold on 
title('14 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 50]);
xlim([0 250]);
ypos1= dat_14deg_100ms(25:48,2) - dat_14deg_100ms(1:24,2);
p1 = shadedErrorBar(dat_14deg_100ms(1:24,1), dat_14deg_100ms(1:24,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_14deg_100ms(1:24,1), dat_14deg_100ms(1:24,2), '-r', 'LineWidth', 2); 
plot(time4+offset, g*deg14_100ms(1800, :), 'k', 'LineWidth', 3);
hold off
% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
han.FontSize = 12;
ylabel(han,'Activity (spikes/s)', 'FontSize', 16, 'FontWeight', 'normal', 'FontName', 'Arial');
xlabel(han,'Time (ms)', 'FontSize', 14, 'FontWeight', 'normal', 'FontName', 'Arial');

figure(3);
sizedeg = [5, 8, 11, 14];
hold on
title('Stimulus duration 40 ms');
xlabel('Stimulus size (deg)', 'FontSize', 14, 'FontWeight', 'normal', 'FontName', 'Arial');
ylabel('Avg activity',  'FontSize', 14, 'FontWeight', 'normal', 'FontName', 'Arial');
plot(sizedeg, [mean_5deg_40ms, mean_8deg_40ms, mean_11deg_40ms, mean_14deg_40ms], 'b', 'LineWidth', 2);
%plot(sizedeg, [mean(deg5_40ms(1800, :)), mean(deg8_40ms(1800, :)), mean(deg11_40ms(1800, :)), mean(deg14_40ms(1800, :))], 'c', 'LineWidth', 2);
plot(sizedeg, [m1, m2, m3, m4], 'g', 'LineWidth', 2);
hold off

%% Stimulus duration --> 40 ms (Non-Surround Suppressed)

% 5 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_5deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_5deg_time_nss.mat');

% Exp. data
dat_5deg_40ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_40ms_nss_.csv');

deg5_40ms_nss = deg5_40ms_nss - 36;
deg5_40ms_nss(deg5_40ms_nss < 0.0) = 0.0;
time1_nss = scale*deg5_40ms_times_nss;

% 8 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_8deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_8deg_time_nss.mat');

% Exp. data
dat_8deg_40ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg8_40ms_nss_.csv');

deg8_40ms_nss = deg8_40ms_nss - 36;
deg8_40ms_nss(deg8_40ms_nss < 0.0) = 0.0;
time2_nss = scale*deg8_40ms_times_nss;

% 11 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_11deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_11deg_time_nss.mat');

% Exp. data
dat_11deg_40ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg11_40ms_nss_.csv');

deg11_40ms_nss = deg11_40ms_nss - 36;
deg11_40ms_nss(deg11_40ms_nss < 0.0) = 0.0;
time3_nss = scale*deg11_40ms_times_nss;

% 14 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_14deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_40ms_14deg_time_nss.mat');

% Exp. data
dat_14deg_40ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg14_40ms_nss_.csv');

deg14_40ms_nss = deg14_40ms_nss - 36;
deg14_40ms_nss(deg14_40ms_nss < 0.0) = 0.0;
time4_nss = scale*deg14_40ms_times_nss;

fig = figure(5);
gain = 0.52;
offset = 50;
subplot(1,4,1);
hold on
title({'Stimulus duration 40 ms ------- Non-surround suppressed'; '5 deg'}, 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
dat_5deg_40ms_nss_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_40ms_nss_std.csv');
ypos1 = dat_5deg_40ms_nss_std(:,2) - dat_5deg_40ms_nss(:,2);
% plot(dat_5deg_40ms_nss(:,1), dat_5deg_40ms_nss(:,2), 'LineWidth', 2);
plot(time1_nss+offset, gain*deg5_40ms_nss(1800, :), 'k', 'LineWidth', 3);
p1 = shadedErrorBar(dat_5deg_40ms_nss(:,1), dat_5deg_40ms_nss(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_5deg_40ms_nss(:,1), dat_5deg_40ms_nss(:,2), '-r', 'LineWidth', 2); 
hold off
subplot(1,4,2);
hold on 
title('8 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
dat_8deg_40ms_nss_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg8_40ms_nss_std.csv');
ypos1 = dat_8deg_40ms_nss_std(:,2) - dat_8deg_40ms_nss(:,2);
% plot(dat_5deg_40ms_nss(:,1), dat_5deg_40ms_nss(:,2), 'LineWidth', 2);
plot(time2_nss+offset, gain*deg8_40ms_nss(1800, :), 'k', 'LineWidth', 3);
p1 = shadedErrorBar(dat_8deg_40ms_nss(:,1), dat_8deg_40ms_nss(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_8deg_40ms_nss(:,1), dat_8deg_40ms_nss(:,2), '-r', 'LineWidth', 2); 
hold off
subplot(1,4,3);
hold on 
title('11 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
dat_11deg_40ms_nss_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg11_40ms_nss_std.csv');
ypos1 = dat_11deg_40ms_nss_std(:,2) - dat_11deg_40ms_nss(:,2);
% plot(dat_5deg_40ms_nss(:,1), dat_5deg_40ms_nss(:,2), 'LineWidth', 2);
plot(time3_nss+offset, gain*deg11_40ms_nss(1800, :), 'k', 'LineWidth', 3);
p1 = shadedErrorBar(dat_11deg_40ms_nss(:,1), dat_11deg_40ms_nss(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_11deg_40ms_nss(:,1), dat_11deg_40ms_nss(:,2), '-r', 'LineWidth', 2);
hold off
subplot(1,4,4);
hold on 
title('14 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
dat_14deg_40ms_nss_std = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg14_40ms_nss_std.csv');
ypos1 = dat_14deg_40ms_nss_std(:,2) - dat_14deg_40ms_nss(:,2);
% plot(dat_5deg_40ms_nss(:,1), dat_5deg_40ms_nss(:,2), 'LineWidth', 2);
plot(time4_nss+offset, gain*deg14_40ms_nss(1800, :), 'k', 'LineWidth', 3);
p1 = shadedErrorBar(dat_14deg_40ms_nss(:,1), dat_14deg_40ms_nss(:,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_14deg_40ms_nss(:,1), dat_14deg_40ms_nss(:,2), '-r', 'LineWidth', 2);
hold off
% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Activity (spikes/s)', 'FontSize', 16, 'FontWeight', 'normal', 'FontName', 'Arial');
xlabel(han,'Time (ms)', 'FontSize', 14, 'FontWeight', 'normal', 'FontName', 'Arial');

figure(6);
hold on
plot(deg5_40ms_times_nss, deg5_40ms_nss(1800, :));
%plot(deg8_40ms_times_nss, deg8_40ms_nss(1800, :));
plot(deg11_40ms_times_nss, deg11_40ms_nss(1800, :));
plot(deg14_40ms_times_nss, deg14_40ms_nss(1800, :));


%% Stimulus duration --> 100 ms (Non-Surround Suppressed)

% 5 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_5deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_5deg_time_nss.mat');

% Exp. data
dat_5deg_100ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg5_100ms_nss_.csv');

deg5_100ms_nss = deg5_100ms_nss - 36;
deg5_100ms_nss(deg5_100ms_nss < 0.0) = 0.0;
time1_nss = scale*deg5_100ms_times_nss;

% 8 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_8deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_8deg_time_nss.mat');

% Exp. data
dat_8deg_100ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg8_100ms_nss_.csv');


deg8_100ms_nss = deg8_100ms_nss - 36;
deg8_100ms_nss(deg8_100ms_nss < 0.0) = 0.0;
time2_nss = scale*deg8_100ms_times_nss;

% 11 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_11deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_11deg_time_nss.mat');

% Exp. data
dat_11deg_100ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg11_100ms_nss_.csv');


deg11_100ms_nss = deg11_100ms_nss - 36;
deg11_100ms_nss(deg11_100ms_nss < 0.0) = 0.0;
time3_nss = scale*deg11_100ms_times_nss;

% 14 deg nss
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_14deg_nss.mat');
load('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/normalization_100ms_14deg_time_nss.mat');

% Exp. data
dat_14deg_100ms_nss = csvread('/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Motion Dynamics/MotionDyn1/dataExp/deg14_100ms_nss_.csv');


deg14_100ms_nss = deg14_100ms_nss - 36;
deg14_100ms_nss(deg14_100ms_nss < 0.0) = 0.0;
time4_nss = scale*deg14_100ms_times_nss;

fig = figure(7);
gain = 0.52;
offset = 50.;
subplot(1,4,1);
hold on
title({'Stimulus duration 100 ms ------- Non-surround suppressed'; '5 deg'},  'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
ypos1= dat_5deg_100ms_nss(23:44,2) - dat_5deg_100ms_nss(1:22,2);
p1 = shadedErrorBar(dat_5deg_100ms_nss(1:22,1), dat_5deg_100ms_nss(1:22,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_5deg_100ms_nss(1:22,1), dat_5deg_100ms_nss(1:22,2), '-r', 'LineWidth', 2); 
plot(time1_nss+offset, gain*deg5_100ms_nss(1800, :), 'k', 'LineWidth', 3);
hold off
subplot(1,4,2);
hold on 
title('8 deg', 'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
ypos1= dat_8deg_100ms_nss(18:34,2) - dat_8deg_100ms_nss(1:17,2);
p1 = shadedErrorBar(dat_8deg_100ms_nss(1:17,1), dat_8deg_100ms_nss(1:17,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_8deg_100ms_nss(1:17,1), dat_8deg_100ms_nss(1:17,2), '-r', 'LineWidth', 2); 
plot(time2_nss+offset, gain*deg8_100ms_nss(1800, :), 'k', 'LineWidth', 3);
hold off
subplot(1,4,3);
hold on 
title('11 deg' ,'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
ypos1= dat_11deg_100ms_nss(21:40,2) - dat_11deg_100ms_nss(1:20,2);
p1 = shadedErrorBar(dat_11deg_100ms_nss(1:20,1), dat_11deg_100ms_nss(1:20,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_11deg_100ms_nss(1:20,1), dat_11deg_100ms_nss(1:20,2), '-r', 'LineWidth', 2); 
plot(time3_nss+offset, gain*deg11_100ms_nss(1800, :), 'k', 'LineWidth', 3);
hold off
subplot(1,4,4);
hold on 
title('14 deg',  'FontSize', 15, 'FontWeight', 'normal', 'FontName', 'Arial');
ylim([0 40]);
xlim([0 250]);
ypos1= dat_14deg_100ms_nss(20:38,2) - dat_14deg_100ms_nss(1:19,2);
p1 = shadedErrorBar(dat_14deg_100ms_nss(1:19,1), dat_14deg_100ms_nss(1:19,2), ypos1, 'lineprops','-r','patchSaturation',0.33);
p2 = plot(dat_14deg_100ms_nss(1:19,1), dat_14deg_100ms_nss(1:19,2), '-r', 'LineWidth', 2); 
plot(time4_nss+offset, gain*deg14_100ms_nss(1800, :), 'k', 'LineWidth', 3);
hold off
% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Activity (spikes/s)', 'FontSize', 16, 'FontName', 'Arial');
xlabel(han,'Time (ms)', 'FontSize', 14, 'FontName', 'Arial');