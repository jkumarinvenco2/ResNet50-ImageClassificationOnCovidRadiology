outputFolder = fullfile('COVID-19Dataset');
rootFolder = fullfile(outputFolder,  'COVID-19_Radiography_Dataset');

categories = {'COVID','Lung_Opacity', 'Normal', 'Viral Pneumonia'};

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});

imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds);

%COVID
COVID = find(imds.Labels == 'COVID', 1);

%Lung_Opacity
Lung_Opacity = find(imds.Labels == 'Lung_Opacity', 1);

%Normal
Normal = find(imds.Labels == 'Normal', 1);

%Viral Pneumonia
Viral_Pneumonia = find(imds.Labels == 'Viral Pneumonia', 1);

% figure
% subplot(2,2,1);
% imshow(readimage(imds, airplanes));
% subplot(2,2,2);
% imshow(readimage(imds, ferry));
% subplot(2,2,3)
% imshow(readimage(imds, laptop));

%ResNet50
net = resnet50();
%AlexNet

% figure
% plot(net);
% title('Architecture of ResNet-50');
% set(gca, 'YLim', [150, 170]);

net.Layers(1);
net.Layers(end);
% 
numel(net.Layers(end).ClassNames);
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
% 
imageSize = net.Layers(1).InputSize;
% 
augmentedTrainingSet = augmentedImageDatastore(imageSize, ...
     trainingSet, 'ColorPreprocessing', 'gray2rgb');
% 
augmentedTestSet = augmentedImageDatastore(imageSize, ...
     testSet, 'ColorPreprocessing', 'gray2rgb');
% 
% 
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
% 
% figure
% montage(w1)
% title ('First Convolutional Layer Weight');
% 
featureLayer = 'fc1000';
trainingFeaatures = activations(net, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32 , 'OutputAs', 'Columns');
% 
trainingLables = trainingSet.Labels;
classifier = fitcecoc(trainingFeaatures, ...
    trainingLables, 'Learner', 'Linear', 'Coding', 'onevsall', ...
    'ObservationsIn', 'Columns');
% 
testFeaatures = activations(net, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32 , 'OutputAs', 'Columns');

predictLabels = predict(classifier, testFeaatures, 'ObservationsIn', 'columns');

testLabels = testSet.Labels;
% 
confMat = confusionmat(testLabels, predictLabels);

confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));
%sum(confMat, 2)
% 
mean(diag(confMat));
% % 
newImage = imread(fullfile('test102.png'));
ds = augmentedImageDatastore(imageSize, ...
     newImage, 'ColorPreprocessing', 'gray2rgb');
% % 
imageFeatures = activations(net, ...
     ds, featureLayer, 'MiniBatchSize', 32 , 'OutputAs', 'Columns');
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
% 
sprintf("The loaded image belongs to %s class", label)
% % 
