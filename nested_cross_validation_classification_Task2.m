clear; close all; clc;
% This code is using nested cross validation with inner fold classification for RBF Kernel Function.  The values of C range from $2^{-5}$ to $2^{15}$ increasing in steps of 2. The values of sigma range from 0.1 to 2 with a 0.1 increase at each step.
M=readtable('data1.csv');
features=12;
allData = normalize(table2array(M(1:200,1:features)));
targets = normalize(table2array(M(1:200,features+1)));
length(targets)
featSize = size(allData, 2);
kFolds = 10;     %  number of folds
bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
    'C', NaN, 'Score', Inf,'Sigma',NaN);     

kIdx = crossvalind('Kfold', length(targets), kFolds);
for k = 1:kFolds
    trainData = allData(kIdx~=k, :);
    trainTarg = targets(kIdx~=k);
    testData = allData(kIdx==k, :);
    testTarg = targets(kIdx==k);
      bestFeatCombo = struct('SVM', NaN, 'C', NaN);
      bestCScore = inf;
      bestCSigmaScore=inf;
      bestScore = inf;
      bestC = NaN;
      bestCSVM = NaN;
          for sigma = 0.1:0.1:2.0
            gridC = 2.^(-5:2:15);
            for C = gridC
                [C,sigma]
                anSVMModel = fitcsvm(trainData, trainTarg,'KernelFunction', 'RBF', 'KernelScale', sigma,'BoxConstraint', C);
                L = loss(anSVMModel,testData, testTarg);
                Y_hat=predict(anSVMModel,testData);
                test_inaccuracy_for_iter = sum(Y_hat ~= testTarg)/length(testTarg)*100;
                if test_inaccuracy_for_iter < bestCScore        % saving best SVM on parameter        bestCScore = test_inaccuracy_for_iter;                  
                    bestCScore = L;      % selection
                    bestC = C;
                    bestCsigma = sigma;
                    bestCCSVM = anSVMModel;

                    
                end
            end 
            if bestCScore < bestCSigmaScore        % saving best SVM on parameter
                    
                    bestCSigmaScore = bestCScore;      % selection
                    bestSigmaC = bestC;
                    bestsigma1 = bestCsigma;
                    bestSigmaCSVM = bestCCSVM;

            end
          end
        if bestCSigmaScore < bestScore        % saving best SVM on parameter
             bestScore = bestCSigmaScore;      % selection
              bestC1 = bestSigmaC;
              bestsigma2 = bestsigma1;
              bestCSVM = bestSigmaCSVM;              
        end  
    % saving the best SVM over all folds
    if bestCScore < bestSVM.Score
        bestSVM.SVMModel = bestCSVM;
        bestSVM.C = bestC1;
        bestSVM.Sigma=bestsigma2;
        bestSVM.Score = bestScore
    end
end
SV_percent=sum(bestSVM.SVMModel.IsSupportVector==1)/length(bestSVM.SVMModel.IsSupportVector)*100