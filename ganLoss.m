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

function [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated)

% Calculate the loss for the discriminator network.
lossDiscriminator =  -mean(log(probReal)) -mean(log(1-probGenerated));

% Calculate the loss for the generator network.
lossGenerator = -mean(log(probGenerated));

end