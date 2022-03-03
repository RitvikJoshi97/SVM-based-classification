clear; close all; clc;
% This code is using regression for Linear Kernel Function
% reading in the data
M=readtable('CCPP.xlsx');

% selecting relevant columns of imported data
X=M(:,1:4);
Y=M(:,5);

% normalisiing the data using z-score normalisation
[N_tab,C_tab,S_tab]=normalize(X,'zscore');
N=table2array(N_tab); C=table2array(C_tab); S=table2array(S_tab);
[Ny_tab,Cy_tab,Sy_tab]=normalize(Y,'zscore');
N1=table2array(Ny_tab); C1=table2array(Cy_tab); S1=table2array(Sy_tab);

% method for rescaling Xs
Xnew=zeros(height(X),4);
for i=1:4
Xnew(:,i)=N(:,i).*S(i)+C(i).*ones(height(X(:,1)),1);
end
% check rescaling is correct: compare Xnew(1:10,:); X(1:10,:);

% method for rescaling Ys
Ynew=zeros(height(Y),1);
Ynew(:,1)=N1(:,1).*S1(1)+C1(1)*ones(height(Y(:,1)),1);
% to check rescaling is correct: compare Ynew(1:10,1); Y(1:10,1);

% replace X and Y with normalized data
X=N_tab; Y=Ny_tab;

% generating test and train data
% p = % of training data
p=0.8;
n=size(X,1);
v=zeros(n,1);
v(1:floor(p*n),1)=1;
v=v(randperm(n),1);
inds_train=find(v>0.5);
inds_test=find(v<0.5);
Xtrain=X(inds_train,:);
Ytrain=Y(inds_train,:);
Xtest=X(inds_test,:);
Ytest=Y(inds_test,:);

 % transforming back the X data to use later in plots
 Xtest_unscaled=zeros(height(Xtest),4);
 Xtest_arr=table2array(Xtest);
    for i=1:4
        Xtest_unscaled(:,i)=Xtest_arr(:,i).*S(i)+C(i).*ones(height(Xtest(:,1)),1);
    end

% transforming back the Y data to use later in plots
Ytest_arr=table2array(Ytest);
Ytest_unscaled=zeros(height(Ytest),1);
Ytest_unscaled(:,1)=Ytest_arr(:,1).*S1(1)+C1(1)*ones(height(Ytest(:,1)),1);

% vectors containing single features of test observations
feat1=Xtest_unscaled(:,1);
feat2=Xtest_unscaled(:,2);
feat3=Xtest_unscaled(:,3);
feat4=Xtest_unscaled(:,4);

% vector of epsilon values
Eps=[0.001,0.01,0.1,0.5,1,1.5,2,];

% print graphs of predictions vs true values for a single feature
% and for all epsilon values in our vector Eps

for i=1:length(Eps)
    model_svr =fitrsvm(Xtrain,Ytrain,'Epsilon',Eps(i));
    yhat_svr = predict(model_svr, Xtest);

    % transforming back the Y data to use later in plots
    yhat_svr_unscaled=zeros(height(yhat_svr),1);
    yhat_svr_unscaled(:,1)=yhat_svr.*S1(1)+C1(1)*ones(height(yhat_svr(:,1)),1);
    
    % plots and MSE
    figure
    scatter(feat2,Ytest_unscaled,'red');
    hold on
    scatter(feat2,yhat_svr_unscaled,'blue' );
    test_MSE=(Ytest_unscaled-yhat_svr_unscaled).'*(Ytest_unscaled-yhat_svr_unscaled)/length((Ytest_unscaled))
    title('Plot of V against EP predictions & EP values for epsilon = ', Eps(i))
    legend('True values of EP on test set','Predictions of EP on test set')
    xlabel('V') 
    ylabel('EP') 
    hold off
end
