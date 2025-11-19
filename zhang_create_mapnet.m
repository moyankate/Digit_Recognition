function net = zhang_create_mapnet()
% Initializes a feedforward neural network similar to MNIST classification.
% The network consists of three hidden layers (784 neurons each) and an 
% output layer with 10 neurons (for digit classification 0-9).

% Define network architecture
net.layer_sizes = [784; 784; 784; 10];  
net.n_layers = length(net.layer_sizes);  % Total number of layers

% Define receptive field sizes for feature extraction layers
receptive_fields = [0, 0; 7, 7; 7, 7];  
feature_map_sizes = repmat([28, 28], 3, 1);  

% Determines how many upstream neurons are active for weight initialization
neuron_activation_counts = [0; 49; 49; 1568];  

% Set activation function
net.activation = "relog";

% Preallocate cell arrays for network parameters
param_list = {'W', 'W_', 'W__', 'b', 'b_', 'b__', 'v', 'v_', 'v__', ...
              'activations', 'errors', 'grad_weights', 'grad_biases', 'connectivity'};
for param = param_list
    net.(param{1}) = cell(net.n_layers, 1);
end

% Initialize weights and biases
for l = 2:net.n_layers  
    % Kaiming initialization
    scale_factor = sqrt(2 / neuron_activation_counts(l));  

    if l == 4  % Special case for the output layer 
        prev_size = net.layer_sizes(l - 1) + net.layer_sizes(l - 2);
    else  % Hidden layers  
        prev_size = net.layer_sizes(l - 1);
    end
    
    % Initialize weights and momentum variables
    net.W{l} = scale_factor * randn(net.layer_sizes(l), prev_size);
    net.W_{l} = zeros(size(net.W{l}));
    net.W__{l} = zeros(size(net.W{l}));

    % Initialize biases
    net.b{l} = zeros(net.layer_sizes(l), 1);
    net.b_{l} = zeros(net.layer_sizes(l), 1);
    net.b__{l} = zeros(net.layer_sizes(l), 1);
end

% Define receptive field connections for feature mapping layers (hidden layers)
for l = 2:net.n_layers - 1  
    % Initialize connectivity matrix
    net.connectivity{l} = zeros(size(net.W{l}));  
    ones_block = ones(receptive_fields(l, 1), receptive_fields(l, 2));  

    % Compute mapping for each neuron in the feature map
    lowest_top = feature_map_sizes(l, 1) - receptive_fields(l, 1) + 1;  
    ratio = (lowest_top - 1) / (feature_map_sizes(l, 1) - 1);

    for i = 1:feature_map_sizes(l, 1)  
        top = min(lowest_top, 1 + round((i - 1) * ratio));  
        for j = 1:feature_map_sizes(l, 2)  
            left = min(lowest_top, 1 + round((j - 1) * ratio));  
            
            % Create binary mask defining neuron connections
            field = zeros(feature_map_sizes(l, :));
            field(top:(top + receptive_fields(l, 1) - 1), left:(left + receptive_fields(l, 2) - 1)) = ones_block;
            net.connectivity{l}((i - 1) * feature_map_sizes(l, 1) + j, :) = field(:)';
        end
    end
end

% Fully connect the final layer
net.connectivity{4} = ones(10, 1568);  

% Adam optimizer tracking variables
net.adam_1t = 1;  
net.adam_2t = 1;  

end
