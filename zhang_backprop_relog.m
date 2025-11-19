function net = zhang_backprop_relog(net, output_error)

% Number of samples in the current mini-batch during training
batch_size = size(output_error, 2);  

% Number of layers in the network
n_layers = net.n_layers;  

% Store the final layer’s error signal 
net.delta{n_layers} = output_error;  

% Compute gradients for the last layer
net.dL_dW{n_layers} = net.delta{n_layers} * ...
                      ([net.activations{n_layers - 2}; net.activations{n_layers - 1}])' / batch_size;  
net.dL_db{n_layers} = sum(net.delta{n_layers}, 2) / batch_size;  

% Now, propagate errors backward through the network
for layer_idx = n_layers - 1:-1:2  

    if layer_idx == 3  % Special case for layer 3
        % Compute error signal for layer 3 using ReLog activations derivative
        net.delta{layer_idx} = (net.W{layer_idx + 1}(:, 785:end))' * net.delta{layer_idx + 1} .* ...
                                (max(0, sign(net.v{layer_idx})) .* (1 ./ (1 + net.v{layer_idx})));  

    elseif layer_idx == 2  % Special case for layer 2 since it connects to both layer 3 and layer 4
        % Compute contribution to error from layer 4
        top_section_W4 = net.W{4}(:, 1:784)';  % The part of W{4} directly connected to layer 2
        delta_2_part1 = top_section_W4 * net.delta{4} .* ...
                        (max(0, sign(net.v{layer_idx})) .* (1 ./ (1 + net.v{layer_idx})));  

        % Compute contribution to error from layer 3
        delta_2_part2 = net.W{layer_idx + 1}' * net.delta{layer_idx + 1} .* ...
                        (max(0, sign(net.v{layer_idx})) .* (1 ./ (1 + net.v{layer_idx})));  

        % Combine both parts to get the total error for layer 2
        net.delta{layer_idx} = delta_2_part1 + delta_2_part2;  

    end
    
    % Compute gradients for weights and biases in the hidden layers
    net.dL_dW{layer_idx} = net.delta{layer_idx} * net.activations{layer_idx - 1}' / batch_size;  
    net.dL_db{layer_idx} = sum(net.delta{layer_idx}, 2) / batch_size;  
end

% Compute the error that would be passed back to the input layer.
% This isn't used for updating weights, but it can be useful for debugging.
net.delta{1} = net.W{2}' * net.delta{2};  

end
