clear;clc;close all
nn_10 = en_reg_nn_fun(10);
fprintf('Mean Squared Error for nn_10 is: %f\n', nn_10);

nn_20 = en_reg_nn_fun(20);
fprintf('Mean Squared Error for nn_20 is: %f\n', nn_20);

nn_20_20 = en_reg_nn_fun([20,20]);
fprintf('Mean Squared Error for nn_20_20 is: %f\n', nn_20_20);

nn_20_20_20 = en_reg_nn_fun([20,20,20]);
fprintf('Mean Squared Error for nn_20_20_20 is: %f\n', nn_20_20_20);

nn_50 = en_reg_nn_fun(50);
fprintf('Mean Squared Error for nn_50 is: %f\n', nn_50);

nn_50_50 = en_reg_nn_fun([50, 50]);
fprintf('Mean Squared Error for nn_50_50 is: %f\n', nn_50_50);

nn_100 = en_reg_nn_fun(100);
fprintf('Mean Squared Error for nn_100 is: %f\n', nn_100);

nn_100_20 = en_reg_nn_fun([100, 20]);
fprintf('Mean Squared Error for nn_100_20 is: %f\n', nn_100_20);

meanSqErr = [nn_10, nn_20, nn_20_20, nn_20_20_20, nn_50, nn_50_50, nn_100, nn_100_20];
netArch = {'nn-10', 'nn-20', 'nn-20-20', 'nn-20-20-20', 'nn-50', 'nn-50-50', 'nn-100', 'nn-100-20'};

figure;
barh(meanSqErr);
for i=1:length(netArch)
    text(1.1 * meanSqErr(i), i, num2str(meanSqErr(i), 4), 'FontSize', 12, 'Color', 'blue')
end

axis([0 100 0 9]);
title('Comparison of neural network architectures for Ensemble Boosting Regressor best predictors');
xlabel('Mean Squared Error');
ylabel('Neural Network Architecture');

set(gca, 'YTick', 1:8);
set(gca, 'YTickLabel', netArch);


function [meanSqErr] = en_reg_nn_fun(hiddenLayers)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here

    %% Load the data
    %%
    load nn_reg_en;
    inputs = X_EN;
    targets = Y_EN;

    %--------------------------------------------------------------------------
    % Now do neural network fit using Levenberg-Marquardt backpropagation
    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

    rng(1);
    hiddenLayerSize = hiddenLayers;
    net = fitnet(hiddenLayerSize, trainFcn);

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

    %% Calculate Mean squared Error for test
    %%
    err = testY - testPred;
    meanSqErr = mean(err.^2, 'omitnan');
end

