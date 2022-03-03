clear; close all; clc;
% This code is using nested cross validation with inner fold classification for Polynomial Kernel Function. The values of C range from $2^{-5}$ to $2^{15}$ increasing in steps of 2. Range of q starts from 2 since it is a polynomial, increases in steps of 1 and goes up to 5.
M=readtable('data1.csv');
features=12;
allData = normalize(table2array(M(1:200,1:features)));
targets = normalize(table2array(M(1:200,features+1)));
length(targets)
featSize = size(allData, 2);
kFolds = 10;     %  number of folds
bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
    'C', NaN, 'Score', Inf,'q',NaN);     

kIdx = crossvalind('Kfold', length(targets), kFolds);
for k = 1:kFolds
    trainData = allData(kIdx~=k, :);
    trainTarg = targets(kIdx~=k);
    testData = allData(kIdx==k, :);
    testTarg = targets(kIdx==k);
      bestFeatCombo = struct('SVM', NaN, 'C', NaN);
      bestCScore = inf;
      bestCqScore=inf;
      bestScore = inf;
      bestC = NaN;
      bestCSVM = NaN;
          for q = 2:1:5
            gridC = 2.^(-5:2:15);
            for C = gridC
                anSVMModel = fitcsvm(trainData, trainTarg,'KernelFunction', 'polynomial','PolynomialOrder',q, 'BoxConstraint', C);
                L = loss(anSVMModel,testData, testTarg);
                Y_hat=predict(anSVMModel,testData);
                test_inaccuracy_for_iter = sum(Y_hat ~= testTarg)/length(testTarg)*100;
                if test_inaccuracy_for_iter < bestCScore        % saving best SVM on parameter                  
                    bestCScore = test_inaccuracy_for_iter;      % selection
                    bestC = C;
                    bestCq = q;
                    bestCCSVM = anSVMModel;

                    
                end
            end 
            if bestCScore < bestCqScore        % saving best SVM on parameter
                    
                    bestCqScore = bestCScore;      % selection
                    bestqC = bestC;
                    bestq1 = bestCq;
                    bestqCSVM = bestCCSVM;

            end
          end
        if bestCqScore < bestScore        % saving best SVM on parameter
             bestScore = bestCqScore;      % selection
              bestC1 = bestqC;
              bestq2 = bestq1;
              bestCSVM = bestqCSVM;              
        end  
    % saving the best SVM over all folds
    if bestCScore < bestSVM.Score
        bestSVM.SVMModel = bestCSVM;
        bestSVM.C = bestC1;
        bestSVM.q=bestq2;
        bestSVM.Score = bestScore
    end
end
SV_percent=sum(bestSVM.SVMModel.IsSupportVector==1)/length(bestSVM.SVMModel.IsSupportVector)*100