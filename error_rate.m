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

function [lossFunction] = error_rate(paras)

load augimds; load dlnetGenerator; load dlnetDiscriminator;

numLatentInputs = 100; 
learnRate = paras(1);
gradientDecayFactor = paras(2);
squaredGradientDecayFactor = paras(3);
squaredGradientDecayFactor = 0.999;
executionEnvironment = "auto";
flipFactor = 0.3;
iteration = 0;
miniBatchSize = 64;
augimds.MiniBatchSize = miniBatchSize;

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];
learnRate1 = 0.0002;
gradientDecayFactor1 = 0.5;

% Reset and shuffle datastore.
    reset(augimds);
    augimds = shuffle(augimds);
    
    % Loop over mini-batches.
    while hasdata(augimds)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.
        X = cat(4,data{:,1}{:});
        X = single(X);
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
        % Rescale the images in the range [-1 1].
        X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
        
        % Convert mini-batch of data to dlarray and specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
        dlnetGenerator.State = stateGenerator;
        
        
        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRate1, gradientDecayFactor1, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

% Convert the discriminator outputs to probabilities.
probGenerated = sigmoid(dlYPredGenerated);
probReal = sigmoid(dlYPred);

lossFunction = mean(1-probGenerated);
lossFunction = double(gather(extractdata(lossFunction)));
% Calculate the score of the discriminator.
scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);


% Calculate the score of the generator.
scoreGenerator = mean(probGenerated);

% Randomly flip a fraction of the labels of the real images.
numObservations = size(probReal,4);
idx = randperm(numObservations,floor(flipFactor * numObservations));

% Flip the labels
probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

lossDiscriminator =  -mean(log(probReal)) -mean(log(1-probGenerated));
lossDiscriminator = double(gather(extractdata(lossDiscriminator)));
% Calculate the loss for the generator network.
% lossGenerator = -mean(log(probGenerated));
% lossGenerator = double(gather(extractdata(lossGenerator)));
end
        
        