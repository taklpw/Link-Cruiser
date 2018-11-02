clear;
clc;
close all;

%three_d_array = csvread('stationary2.txt');
%allColors =  three_d_array(:, 4:6)/255;
%cloud = pointCloud(three_d_array(:, 1:3), 'color', allColors);
cloud = pcread('kellysroom.pcd');

roi = [-2,2;-2, 4;-2,1];
indices = findPointsInROI(cloud,roi);
cleaned_cloud = select(cloud ,indices);

maxAngularDistance = 5;

%roi = [-inf,0;-1.2, 0;-inf,inf];
%sampleIndices = findPointsInROI(cleaned_cloud,roi);
referenceVector = [0, 1, 0];
[model1,inlierIndices,outlierIndices]  = pcfitplane(cleaned_cloud, 0.05, referenceVector, maxAngularDistance);
remainPtCloud = select(cleaned_cloud ,outlierIndices);
figure;
pcshow(remainPtCloud);
title('ptcloudremains');


[model2,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud, 0.2);
remainPtCloud = select(remainPtCloud ,outlierIndices);
figure;
pcshow(remainPtCloud);
title('ptcloudremains');

[model3,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud, 0.05);
remainPtCloud = select(remainPtCloud ,outlierIndices);
figure;
pcshow(remainPtCloud);
title('ptcloudremains');


[model4,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud, 0.05);
remainPtCloud = select(remainPtCloud ,outlierIndices);
figure;
pcshow(remainPtCloud);
title('ptcloudremains');

figure;
pcshow(cleaned_cloud, 'MarkerSize', 60);
xlabel('x');
ylabel('y');
zlabel('z');
hold on;
plot1=plot(model2);
hold on;
plot2=plot(model3);
hold on;
plot3=plot(model4);

syms x y z
% eqn1 = (model1.Parameters(1))*x + (model1.Parameters(2))*y + (model1.Parameters(3))*z + (model1.Parameters(4))== 0;
% eqn2 = (model3.Parameters(1))*x + (model3.Parameters(2))*y + (model3.Parameters(3))*z + (model3.Parameters(4))== 0;
% eqn3 = z == 100;
% eqn4 = z == 0;
eqn = -(model2.Parameters(1))*x + (model2.Parameters(3))*z + (model2.Parameters(4))== 0;
eqn2 = -(model3.Parameters(1))*x + (model3.Parameters(3))*z + (model3.Parameters(4))== 0;
eqn3 = -(model4.Parameters(1))*x + (model4.Parameters(3))*z + (model4.Parameters(4))== 0;



sol=solve([eqn,eqn2]);
sol2=solve([eqn,eqn3]);

figure;
ez1=ezplot(eqn,[double(sol2.x),double(sol.x)]);
hold on;
ez2=ezplot(eqn2, [double(sol.z), 2]);
hold on;
ez3=ezplot(eqn3, [double(sol2.x), 2]);
set(ez1,'color',[1 0 0]);
set(ez2,'color',[0 1 0]);
set(ez3,'color',[0 0 1]);
wall_length=pdist([double(sol2.x),double(sol2.z);double(sol.x),double(sol.z)], 'euclidean');
text((double(sol2.x)+double(sol.x))/2,(double(sol2.z)+double(sol.z))/2.2,sprintf('Wall length %f', wall_length));
legend('back wall', 'right wall', 'left wall');
title('floor map (metres)');
