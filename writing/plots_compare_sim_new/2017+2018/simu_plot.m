clear all
% close all

%file = '2017_month.csv'
file = 'month.csv'
%file = '2017_week.csv'
%file = '2018_week.csv'


T = readtable(file);

simu_names = table2cell(T(:,1));
score_tot = table2array(T(:,2))/797.16;
score = table2array(T(:,3));

simu_names_small = simu_names
%for i=1:length(simu_names)
%    simu_names_small{i} = simu_names{i}(69:end);
%end



colormap_linda2 =[         0         0         0
    0.5020         0    0.5020
    0.2745    0.6392    1.0000
         0         0    0.7490
    0.8157    0.9216    0.0706
   0.2235    0.7922    0.0588
         0    0.4235         0
    0.9922    0.6941    0.0745
    1.0000         0         0
   0.4667         0         0
    0.5020    0.5020    0.5020]

simu_cd004 = find(~cellfun(@isempty,strfind(simu_names_small, 'advection'))) %01
simu_cd01 = find(~cellfun(@isempty,strfind(simu_names_small, 'justwinds'))) %015
simu_cd015 = find(~cellfun(@isempty,strfind(simu_names_small, '0deg'))) %02
simu_cdrelou = find(~cellfun(@isempty,strfind(simu_names_small, 'randeg'))) %02

simu_cd02 = find(~cellfun(@isempty,strfind(simu_names_small, 'varydeg'))) %03


%simu_0deg = find(~cellfun(@isempty,strfind(simu_names_small, '_0deg')))
%simu_adv = find(~cellfun(@isempty,strfind(simu_names_small, 'advection')))
%simu_justWind = find(~cellfun(@isempty,strfind(simu_names_small, 'justwinds')))
%simu_randdeg = find(~cellfun(@isempty,strfind(simu_names_small, '_randeg_')))
%simu_varydeg = find(~cellfun(@isempty,strfind(simu_names_small, '_varydeg_')))

v = [1,0,1]

figure
scatter(score(simu_cd004),score_tot(simu_cd004),100,'b','x')
hold on
scatter(score(simu_cd01),score_tot(simu_cd01),100,'r','+')
scatter(score(simu_cd015),score_tot(simu_cd015),100,v,'o')
scatter(score(simu_cdrelou),score_tot(simu_cdrelou),100,v,'*')

scatter(score(simu_cd02),score_tot(simu_cd02),100,'b','o','filled')
legend('advection','justwinds','combined 0deg', 'combined randeg', 'combined varydeg')
xlim([0 3.5])
ylabel('% of BB beached')
xlabel('Difference with obs (normalized)')
title(file(1:end-4))
saveas(gcf,['fig_total' file(1:end-4) '.png'])

