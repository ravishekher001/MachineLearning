clear;clc;close all
nn_10 = tb_reg_nn_fun(10);
fprintf('Mean Squared Error for nn_10 is: %f\n', nn_10);


function [meanSqErr] = tb_reg_nn_fun(hiddenLayers)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here

    %% Load the data
    %%
    load nn_reg_tb;
    inputs = X_TB;
    targets = Y_TB;

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

    %% Set up Division of Data for Training, Validation, Testing
    %%
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    %% Train the Network
    %%
    [net,tr] = train(net,inputs,targets);

    %% Test the Network on both training and test sets
    %%
    trainX = X_TB(:, tr.trainInd);
    trainY = Y_TB(:, tr.trainInd);
    testX = X_TB(:, tr.testInd);
    testY = Y_TB(:, tr.testInd);
    trainPred = net(trainX);
    testPred = net(testX);

    % plot error histogram
    plotErrorHistogram(trainY, trainPred, testY, testPred)
    
    %% Calculate Mean squared Error for train and test
    %%
    err_train = trainY - trainPred;
    err_test = testY - testPred;
    meanSqErr_train = mean(err_train.^2, 'omitnan');
    meanSqErr_test = mean(err_test.^2, 'omitnan');
    
    %% Calculate r-square
    % calculate the correlation coefficients for the training and test data 
    % sets with the associated linear fits 
    R_train = corrcoef(trainY,trainPred)
    R_test = corrcoef(testY,testPred)
    r_train=R_train(1,2)
    r_test=R_test(1,2)
    
    % plot scatter plot
    plotScatterDiagram(trainY, trainPred, testY, testPred, r_train, r_test)
    
    meanSqErr = meanSqErr_test
    
end

