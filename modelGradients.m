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

function [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor)


% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

% Convert the discriminator outputs to probabilities.
probGenerated = sigmoid(dlYPredGenerated);
probReal = sigmoid(dlYPred);

% Calculate the score of the discriminator.
scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

% Calculate the score of the generator.
scoreGenerator = mean(probGenerated);

% Randomly flip a fraction of the labels of the real images.
numObservations = size(probReal,4);
idx = randperm(numObservations,floor(flipFactor * numObservations));

% Flip the labels
probReal(:,:,:,idx) = 1-probReal(:,:,:,idx);

% % % SearchAgents_no = 30;
% % % Max_iteration = 3;
% % % lb = .01;
% % % ub = 0.5;
% % % dim= 2;
% % % 
% % % [Best_score,Best_pos,WOA_cg_curve]=WOA(SearchAgents_no,Max_iteration,lb,ub,dim,@ganLoss);
% % % 
% % % % Calculate the GAN loss.
   [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated);
% % % % % % % 
% % % % % % % lossGenerator = Best_pos(1,1);
% % % % % % % lossDiscriminator = Best_pos(1,2);
% % % % 
% % % %     x0 = [0.01, 0.01]; 
% % % %    
% % % % %     options.InitialSwarmMatrix = x0;
% % % %     options = optimoptions('particleswarm','InitialSwarmMatrix',x0, 'MaxIterations', 2, 'MaxTime', 2);
% % % %     lb =  [0.01, 0.01];
% % % %     ub = [0.1, 0.5]; 
% % % %     
% % % %     [lossGenerator, lossDiscriminator] = particleswarm(@ganLoss,2,lb,ub,options);

% [lossGenerator, lossDiscriminator] = Best_pos(1,2);
% For each network, calculate the gradients with respect to the loss.
gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end

