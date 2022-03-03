clear; close all; clc;
% This code is using nested cross validation with outer fold classification for Polynomial Kernel Function
M=readtable('data1.csv');
features=12;
N=normalize(M);
X1=table2array(M(:,1:features));
Y1=table2array(M(:,features+1));
[N_tab,C_tab,S_tab]= normalize(X1,'zscore');
[Ny_tab,Cy_tab,Sy_tab]= normalize(Y1,'zscore');
X=N_tab;
Y=Ny_tab;
cp=classperf(Y1)
allData = X(:,1:features);
targets = Y(:,1);
length(targets)

kouterFolds = 10;
bestSVM_outerfold2(kouterFolds) =  struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'q',NaN);  

kouterIdx = crossvalind('Kfold', length(targets), kouterFolds);
    for k = 1:kouterFolds
        k
        kfolds=2;
        trainData = allData(kouterIdx~=k, :);
        trainTarg = targets(kouterIdx~=k);
        testData = allData(kouterIdx==k, :);
        testTarg = targets(kouterIdx==k);
        
        bestSVM_innerfold1 =  struct('SVMModel', NaN, ...     % this is to store the best SVM
        'C', NaN, 'Score', Inf,'q',NaN);

        bestSVM_innerfold1 = nested_cross_valid(kfolds,trainData,trainTarg)
        
        ["outerfolds: ",kouterFolds]
       
        bestSVM_outerfold2(k).SVMModel = bestSVM_innerfold1.SVMModel;
        bestSVM_outerfold2(k).C = bestSVM_innerfold1.C;
        bestSVM_outerfold2(k).q=bestSVM_innerfold1.q;
        bestSVM_outerfold2(k).Score = bestSVM_innerfold1.Score
    end

    [train,test] = crossvalind('HoldOut',length(Y),0.2);
   

for i=1:kouterFolds

    anSVMModel = fitcsvm(X(train), Y(train),'KernelFunction', 'polynomial','PolynomialOrder',bestSVM_outerfold2(i).q, 'BoxConstraint', bestSVM_outerfold2(i).C);
    Y_hat=predict(anSVMModel,X(test));
    Y_hat_unscale=Y_hat*Sy_tab(1)+Cy_tab*ones(height(Y_hat(:,1)),1);
    classperf(cp,Y_hat_unscale,test)
end

function bestSVM =nested_cross_valid(kFolds,allData,targets)
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
end 
