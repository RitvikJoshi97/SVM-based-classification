clear; close all; clc;
% This code is using inner fold regression for RBF Kernel Function
M=readtable('CCPP.xlsx');
features=4;
allData = normalize(table2array(M(1:1000,1:features)));
targets = normalize(table2array(M(1:1000,features+1)));
length(targets)
featSize = size(allData, 2);
kFolds = 10;     %  number of folds
bestSVM = struct('SVMModel', NaN, ...     % this is to store the best SVM
    'C', NaN, 'Score', Inf,'Sigma',NaN,'Episilon',NaN);     

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
      bestRSVM = NaN;
      for epsilon = 0.1:0.1:1.2    % 
          for sigma = 0.1:0.1:2.0
            gridC = 2.^(-5:2:9);
            for C = gridC
                epsilon
                anSVMModel = fitrsvm(trainData, trainTarg,'KernelFunction', 'RBF', 'KernelScale', sigma,'BoxConstraint', C,'Epsilon',epsilon);
                L = loss(anSVMModel,testData, testTarg);
                Y_hat=predict(anSVMModel,testData);
                test_MSE=(Y_hat-testTarg).'*(Y_hat-testTarg)/length((Y_hat));
                sqrt(test_MSE)
                if test_MSE < bestCScore        % saving best SVM on parameter                  
                    bestCScore = test_MSE; 
                    bestC = C;
                    bestCsigma = sigma;
                    bestCRSVM = anSVMModel;
                    bestCepsilon= epsilon;
                    
                end
            end 
            if bestCScore < bestCSigmaScore        % saving best SVM on parameter
                    
                    bestCSigmaScore = bestCScore;      % selection
                    bestSigmaC = bestC;
                    bestsigma1 = bestCsigma;
                    bestSigmaRSVM = bestCRSVM;
                    bestSigmaepsilon= bestCepsilon;
            end
          end
        if bestCSigmaScore < bestScore        % saving best SVM on parameter
             bestScore = bestCSigmaScore;      % selection
              bestC1 = bestSigmaC;
              bestsigma2 = bestsigma1;
              bestRSVM = bestSigmaRSVM;
              bestepsilon= bestSigmaepsilon;
              
        end  
      end
    % saving the best SVM over all folds
    if bestScore < bestSVM.Score
        bestSVM.SVMModel = bestRSVM;
        bestSVM.C = bestC1;
        bestSVM.Sigma=bestsigma2;
        bestSVM.Episilon = bestepsilon;
        bestSVM.Score = bestScore
    end
end
SV_percent=sum(bestSVM.SVMModel.IsSupportVector==1)/length(bestSVM.SVMModel.IsSupportVector)*100