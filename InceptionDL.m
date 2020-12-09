    %_________________________________________________________________________%
    %  Automatic Screening of COVID?19 Using an Optimized Generative
    % Adversarial Network source codes demo V1.0        %
    %                                                                         %
    %  Developed in MATLAB R2018b                                             %
    %                                                                         %
    %  Author and programmer: Tripti Goel                                     %
    %                                                                         %
    %         e-Mail: triptigoel83@gmail.com                                  %
    %                 triptigoel@ece.nits.ac.in                               %
    %                                                                         %
    %       Homepage: http://www.nits.ac.in/departments/ece/ece.php           %
    %                                                                         %
    %  Main paper:  Tripti Goel R. Murugan· Seyedali Mirjalili and  ·
    %               Deba Kumar Chakrabartty3                               %
    %               Automatic Screening of COVID?19 Using an Optimized Generative
    %               Adversarial Network%
    %               Cognitive Computation
    %               https://doi.org/10.1007/s12559-020-09785-7
    %
    %                                                                         %
    %_________________________________________________________________________%

    digitDatasetPath = fullfile('C:\Users\MIPLAB\Documents\Covid\Database\WOAGeneratedImageDatabase');

    imds = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    tbl = countEachLabel(imds);

    minSetCount = min(tbl{:,2});
    net = inceptionv3();

    no_person = 2;

    [trainingSet, testSet] = splitEachLabel(imds, 0.9, 'randomize');

    inputSize = net.Layers(1).InputSize;


    if isa(net,'SeriesNetwork')
        lgraph = layerGraph(net.Layers);
    else
        lgraph = layerGraph(net);
    end

    layersToRemove = {
        'predictions'
        'predictions_softmax'
        'ClassificationLayer_predictions'
        };

    lgraph = removeLayers(lgraph, layersToRemove); % Remove the the last 3 layers.

    % Specify the number of classes the network should classify.
    numClassesPlusBackground = 2;

    % Define new classfication layers
    newLayers = [
        fullyConnectedLayer(numClassesPlusBackground, 'Name', 'rcnnFC')
        softmaxLayer('Name', 'rcnnSoftmax')
        classificationLayer('Name', 'rcnnClassification')
        ];

    % Add new layers
    lgraph = addLayers(lgraph, newLayers);

    % Connect the new layers to the network.
    lgraph = connectLayers(lgraph, 'avg_pool', 'rcnnFC');

    % % % numClasses = numel(categories(imdsTrain.Labels));
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),trainingSet,'ColorPreprocessing', 'gray2rgb');
    augmentedTestSet = augmentedImageDatastore(inputSize(1:2),testSet, 'ColorPreprocessing', 'gray2rgb');
    YValidation = testSet.Labels;

    save augmentedTrainingSet augmentedTrainingSet;
    save  augmentedTestSet  augmentedTestSet;
    save YValidation YValidation


    options = trainingOptions('sgdm', ...
        'MiniBatchSize',32, ...
        'MaxEpochs',10, ...
        'InitialLearnRate',1e-4, ...
        'Shuffle','every-epoch', ...
        'ValidationData',augmentedTestSet, ...
        'ValidationFrequency',30, ...
        'Verbose',false, ...
        'Plots','training-progress');

    net = trainNetwork(augmentedTrainingSet,lgraph,options);



    [YPred,Scores] = classify(net,augmentedTestSet);
    accuracy = sum(YPred == YValidation)/numel(YValidation);

    no_img_p_s_train = 585;
    dimTest  = 130;
    fprintf('Creating the target matrix of TestData for calculating accuracy(Performance)\n');
    no_img_p_s_test = 649-no_img_p_s_train;

    TestTargets = zeros(no_person, dimTest);
    for j = 1:no_person
        for k = 1:no_img_p_s_test
            TestTargets(j,((j-1)*no_img_p_s_test + k)) = 1;
        end
    end

    [x, YValidation_Expected]=max(TestTargets);
    [x, label_index_actual]=max(Scores');

    plotroc(TestTargets, Scores')

    confMat = confusionmat(YValidation, YPred);
    confMat1 = bsxfun(@rdivide,confMat,sum(confMat,2));
    plotConfMat(confMat, {'COVID', 'Normal'});
    EVAL = Evaluate(YValidation_Expected, label_index_actual)






