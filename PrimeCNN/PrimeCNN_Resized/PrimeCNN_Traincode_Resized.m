inputSize = [128, 128, 3];

% Create a network
lgraph = layerGraph;
layers = [
    imageInputLayer(inputSize,"Name","imageinput")
    groupedConvolution2dLayer([3 3],32,"channel-wise","Name","groupedconv1")
    transposedConv2dLayer([3 3],8,"Name","transposed-conv1","Cropping","same")
    convolution2dLayer([3 3],32,"Name","conv1")
    reluLayer("Name","relu1")
    batchNormalizationLayer("Name","BatchNorm1")
    maxPooling2dLayer([2 2],"Name","maxpool1","Padding","same","Stride",[2 2])
    transposedConv2dLayer([3 3],8,"Name","transposed-conv2","Cropping","same")
    convolution2dLayer([3 3],8,"Name","conv2")
    reluLayer("Name","relu2")
    batchNormalizationLayer("Name","BatchNorm2")
    convolution2dLayer([1 1],8,"Name","conv3")
    reluLayer("Name","relu3")
    batchNormalizationLayer("Name","BatchNorm3")
    groupedConvolution2dLayer([3 3],16,"channel-wise","Name","groupedconv2")
    transposedConv2dLayer([3 3],32,"Name","transposed-conv3","Cropping","same")
    convolution2dLayer([3 3],32,"Name","conv4","Padding","same")
    batchNormalizationLayer("Name","BatchNorm4")
    maxPooling2dLayer([2 2],"Name","maxpool2","Stride",[2 2])
    fullyConnectedLayer(128,"Name","fc1")
    dropoutLayer(0.7,"Name","dropout1")
    fullyConnectedLayer(64,"Name","fc2")
    dropoutLayer(0.7,"Name","dropout2")
    fullyConnectedLayer(10,"Name","fc3")
    classificationLayer("Name","classoutput")];

lgraph = addLayers(lgraph, layers);
analyzeNetwork(lgraph); 
figure
plot(lgraph)

%load the data
data = 'F:\Shell_Intern\Shell_Project\Data images\Resize_Preprocess';
Datasetpath = fullfile(data);
imds = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

% Split the Tomato leaf Diseases images from Data in the ratio 80:20 and normal images also 
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

% Resize the images to the input size of first layer
augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', 5, ... 
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-5, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the network
trainedModelprimeresized = trainNetwork(augimdsTrain, layers, options);

% Save the trained model to a MAT file
save(fullfile('F:\Shell_Intern\Shell_Project\TrainData_Matlab\trainedModelprimeresized'), 'trainedModelprimeresized');