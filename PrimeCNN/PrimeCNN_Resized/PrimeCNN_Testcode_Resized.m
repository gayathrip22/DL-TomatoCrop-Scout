load('F:\Shell_Intern\Shell_Project\TrainData_Matlab\trainedModelprimeresized.mat');

data = 'F:\Shell_Intern\Shell_Project\Data images\Resize_Preprocess';
Datasetpath = fullfile(data);
imds = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

%Split the Tomato leaf Diseases images from Data in the ratio 80:20 and normal images also 
[imds_TS80, imds_TS20] = splitEachLabel(imds, 0.8,'Include','Tomato__Target_Spot');
[imds_MV80, imds_MV20] = splitEachLabel(imds, 0.8,'Include', 'Tomato__Tomato_mosaic_virus');
[imds_YL80, imds_YL20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato__Tomato_YellowLeaf__Curl_Virus');
[imds_BS80, imds_BS20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato_Bacterial_spot');
[imds_EB80, imds_EB20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato_Early_blight');
[imds_H80, imds_H20]   = splitEachLabel(imds, 0.8, 'Include', 'Tomato_healthy');
[imds_LB80, imds_LB20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato_Late_blight');
[imds_LM80, imds_LM20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato_Leaf_Mold');
[imds_SL80, imds_SL20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato_Septoria_leaf_spot');
[imds_SS80, imds_SS20] = splitEachLabel(imds, 0.8, 'Include', 'Tomato_Spider_mites_Two_spotted_spider_mite');

% Final Training set
FinalTrain = imageDatastore(cat(1,imds_TS80.Files,imds_MV80.Files,imds_YL80.Files,imds_BS80.Files,imds_EB80.Files,imds_H80.Files,imds_LB80.Files,imds_LM80.Files,imds_SL80.Files,imds_SS80.Files));
FinalTrain.Labels = cat(1,imds_TS80.Labels,imds_MV80.Labels,imds_YL80.Labels,imds_BS80.Labels,imds_EB80.Labels,imds_H80.Labels,imds_LB80.Labels,imds_LM80.Labels,imds_SL80.Labels,imds_SS80.Labels);

% Final Testing set
FinalTest = imageDatastore(cat(1,imds_TS20.Files,imds_MV20.Files,imds_YL20.Files,imds_BS20.Files,imds_EB20.Files,imds_H20.Files,imds_LB20.Files,imds_LM20.Files,imds_SL20.Files,imds_SS20.Files));
FinalTest.Labels = cat(1,imds_TS20.Labels,imds_MV20.Labels,imds_YL20.Labels,imds_BS20.Labels,imds_EB20.Labels,imds_H20.Labels,imds_LB20.Labels,imds_LM20.Labels,imds_SL20.Labels,imds_SS20.Labels);


% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain);

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest);

% Resize the images to the input size of first layer
inputSize = [128,128,3];
augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

layer = 'fc3';
featuresTrain7 = activations(trainedModelprimeresized,augimdsTrain,layer,'OutputAs','rows');
featuresTest7 = activations(trainedModelprimeresized,augimdsTest,layer,'OutputAs','rows');

 YTrain = FinalTrain.Labels;
 YTest = FinalTest.Labels;
 
% Classification using SVM classifier
mdl = fitcecoc(featuresTrain7, YTrain);
YPred = predict(mdl, featuresTest7);
accuracy = mean(YPred == YTest);

figure(1);
plotconfusion(YPred, YTest);
[YPred, scores] = classify(trainedModelprimeresized, augimdsTest);

% Accuracy calculation
%YValidation = FinalTest.Labels;
%accuracy = mean(YPred == YValidation);

% Plot confusion matrix
%figure, plotconfusion(YPred, YValidation);

% Calculate confusion matrix
confMat = confusionmat(YPred, YTest);
figure, plotconfusion(YPred, YTest);
% Number of classes
numClasses = size(confMat, 1);

% Initialize variables to store TP, TN, FP, FN for each class
TP = zeros(1, numClasses);
TN = zeros(1, numClasses);
FP = zeros(1, numClasses);
FN = zeros(1, numClasses);

% Loop through each class
for i = 1:numClasses
    % Calculate TP, TN, FP, FN for class i
    TP(i) = confMat(i, i);
    FN(i) = sum(confMat(i, :)) - TP(i);
    FP(i) = sum(confMat(:, i)) - TP(i);
    TN(i) = sum(confMat(:)) - TP(i) - FN(i) - FP(i);
    Recalrate(i) = (TP(i)/(TP(i)+FN(i)));
    Specificity(i) = (TN(i)/(TN(i)+FP(i)));
    precision(i) = (TP(i)/(TP(i)+FP(i)));
    %F1-score = (2*((precision*RecalRate)/(RecalRate+precision)));
    accuracy(i) = (TP(i) + TN(i)) / (TP(i) + TN(i) + FP(i) + FN(i));
    
end

% Calculate accuracy for each class
%Accuracy=sum(accuracy(i))/10;

% Display TP, TN, FP, FN, and accuracy for each class
for i = 1:numClasses
    disp(['Class ', num2str(i)]);
    disp(['True Positives (TP): ', num2str(TP(i))]);
    disp(['True Negatives (TN): ', num2str(TN(i))]);
    disp(['False Positives (FP): ', num2str(FP(i))]);
    disp(['False Negatives (FN): ', num2str(FN(i))]);
    disp(['RecalRate: ', num2str(Recalrate(i))]);
    disp(['Specificity: ', num2str(Specificity(i))]);
    disp(['precision: ', num2str(precision(i))]);
    disp(['Accuracy: ', num2str(accuracy(i))]);
    %disp(['Accuracy: ', num2str(Accuracy)]);
end