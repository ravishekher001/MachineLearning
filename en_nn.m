clear;clc;close all
nn_10 = en_nn_fun(10);
nn_20 = en_nn_fun(20);
nn_20_20 = en_nn_fun([20,20]);
nn_20_20_20 = en_nn_fun([20,20,20]);
nn_50 = en_nn_fun(50);
nn_50_50 = en_nn_fun([50, 50]);
nn_100 = en_nn_fun(100);
nn_100_20 = en_nn_fun([100, 20]);

pctAccurate = [nn_10, nn_20, nn_20_20, nn_20_20_20, nn_50, nn_50_50, nn_100, nn_100_20];
netArch = {'nn-10', 'nn-20', 'nn-20-20', 'nn-20-20-20', 'nn-50', 'nn-50-50', 'nn-100', 'nn-100-20'};

figure;
barh(pctAccurate);
axis([0 100 0 9]);
title('Comparison of neural network architectures for Ensemble Classifier best predictors');
xlabel('Percent Correct Classified');
ylabel('Neural Network Architecture');

set(gca, 'YTick', 1:8);
set(gca, 'YTickLabel', netArch);

function [pctCorrectClassify] = en_nn_fun(hiddenLayers)
% Ensemble Classifer Neural Network function
% hiddenLayers 
% -scalar value indicates 1 layer with specified neurons: eg: en_nn_fun(10)
% -vector value indicates n layers with specified neurons per layer
% eg: en_nn_fun([20,20])

    %% Load the data
    %%
    load nn_ensemble;
    inputs = X_EN;
    targets = Y_EN;
    %% Create a Pattern Recognition Network
    %%
    rng(1);
    hiddenLayerSize = hiddenLayers;
    net = patternnet(hiddenLayerSize);

    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    %% Train the Network
    %%
    [net,tr] = train(net,inputs,targets);

    %% Test the Network
    %%

    testX = X_EN(:, tr.testInd);
    testY = Y_EN(:, tr.testInd);
    testPred = net(testX);

    %plotconfusion(testY, testPred)

    [c,cm] = confusion(testY,testPred);
    fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
    fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
    %plotroc(testY, testPred)

    pctCorrectClassify = 100*(1-c);
end