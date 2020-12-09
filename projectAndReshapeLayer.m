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
%  Main paper:  Tripti Goel R. Murugan� Seyedali Mirjalili and  � 
%               Deba Kumar Chakrabartty3                               %
%               Automatic Screening of COVID?19 Using an Optimized Generative
%               Adversarial Network%
%               Cognitive Computation
%               https://doi.org/10.1007/s12559-020-09785-7
   %
%                                                                         %
%_________________________________________________________________________%

classdef projectAndReshapeLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.
        OutputSize
    end

    properties (Learnable)
        % Layer learnable parameters.
        
        Weights
        Bias
    end
    
    methods
        function layer = projectAndReshapeLayer(outputSize, numChannels, name)
            % Create a projectAndReshapeLayer.
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Project and reshape layer with output size " + join(string(outputSize));
            
            % Set layer type.
            layer.Type = "Project and Reshape";
            
            % Set output size.
            layer.OutputSize = outputSize;
            
            % Initialize fully connect weights and bias.
            fcSize = prod(outputSize);
            layer.Weights = initializeGlorot(fcSize, numChannels);
            layer.Bias = zeros(fcSize, 1, 'single');
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data, specified as a 1-by-1-by-C-by-N 
            %                 dlarray, where N is the mini-batch size.
            % Outputs:
            %         Z     - Output of layer forward function returned as 
            %                 an sz(1)-by-sz(2)-by-sz(3)-by-N dlarray,
            %                 where sz is the layer output size and N is
            %                 the mini-batch size.

            % Fully connect.
            weights = layer.Weights;
            bias = layer.Bias;
            X = fullyconnect(X,weights,bias,'DataFormat','SSCB');
            
            % Reshape.
            outputSize = layer.OutputSize;
            Z = reshape(X, outputSize(1), outputSize(2), outputSize(3), []);
        end
    end
end

function weights = initializeGlorot(numOut, numIn)
% Initialize weights using uniform Glorot.

varWeights = sqrt( 6 / (numIn + numOut) );
weights = varWeights * (2 * rand([numOut, numIn], 'single') - 1);

end