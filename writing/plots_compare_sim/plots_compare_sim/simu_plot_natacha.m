clear all
% close all

file = '2017_month.csv'
%file = '2018_month.csv'
%file = '2017_week.csv'
%file = '2018_week.csv'


T = readtable(file);

simu_names = table2cell(T(:,1));
score_tot = table2array(T(:,2))/800;
score = table2array(T(:,3));

simu_names_small = simu_names
%for i=1:length(simu_names)
%    simu_names_small{i} = simu_names{i}(69:end);
%end

colormap_linda2 =[         0         0         1
   1    0    1
   1    0    0]


simu_cd004 = find(~cellfun(@isempty,strfind(simu_names_small, 'randeg_CD004_'))) %015
simu_cd01 = find(~cellfun(@isempty,strfind(simu_names_small, 'randeg_CD01'))) %015
simu_cd015 = find(~cellfun(@isempty,strfind(simu_names_small, 'randeg_CD015_'))) %015
simu_cd02 = find(~cellfun(@isempty,strfind(simu_names_small, 'randeg_CD02'))) %015
simu_cd03 = find(~cellfun(@isempty,strfind(simu_names_small, '0deg_CD004_'))) %015
simu_cd04 = find(~cellfun(@isempty,strfind(simu_names_small, '0deg_CD01_'))) %015
simu_0deg = find(~cellfun(@isempty,strfind(simu_names_small, '0deg_CD015_')))
simu_adv = find(~cellfun(@isempty,strfind(simu_names_small, '0deg_CD02_')))
simu_justWind = find(~cellfun(@isempty,strfind(simu_names_small, 'varydeg_CD004_')))
simu_randdeg = find(~cellfun(@isempty,strfind(simu_names_small, 'varydeg_CD01_')))
simu_varydeg = find(~cellfun(@isempty,strfind(simu_names_small, 'varydeg_CD015_')))
simu_varydegyo = find(~cellfun(@isempty,strfind(simu_names_small, 'varydeg_CD02_')))

figure
scatter(score(simu_cd004),score_tot(simu_cd004),100, colormap_linda2(1,:,:) ,'x')
hold on
scatter(score(simu_cd01),score_tot(simu_cd01),100,colormap_linda2(1,:,:) ,'+')
scatter(score(simu_cd015),score_tot(simu_cd015),100,colormap_linda2(1,:,:)  ,'o')
scatter(score(simu_cd02),score_tot(simu_cd02),100,colormap_linda2(1,:,:)  ,'*')
scatter(score(simu_cd03),score_tot(simu_cd03),100,colormap_linda2(3,:,:)  ,'x')
scatter(score(simu_cd04),score_tot(simu_cd04),100,colormap_linda2(3,:,:)  ,'+')
scatter(score(simu_0deg),score_tot(simu_0deg),100,colormap_linda2(3,:,:) ,'o')
scatter(score(simu_adv),score_tot(simu_adv),100,colormap_linda2(3,:,:)  ,'*')
scatter(score(simu_justWind),score_tot(simu_justWind),100,colormap_linda2(2,:,:)  ,'x')
scatter(score(simu_randdeg),score_tot(simu_randdeg),100,colormap_linda2(2,:,:)  ,'+')
scatter(score(simu_varydeg),score_tot(simu_varydeg),100,colormap_linda2(2,:,:)  ,'o')
scatter(score(simu_varydegyo),score_tot(simu_varydegyo),100,colormap_linda2(2,:,:)  ,'*')


%legend('left/right CD10%','left/right cay1','CD : 15%','CD : 20%','CD : 30%','CD : 40%')
xlim([0 3.5])
title(file(1:end-4))
%saveas(gcf,['fig_' file(1:end-4) '.png'])