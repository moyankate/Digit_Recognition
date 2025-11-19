clear variables;  

load 'MNIST.mat';

net = zhang_create_mapnet();  % Create the network 

num_batches = 600;    % Number of mini-batches per epoch
batch_size = 100;     % Number of examples per mini-batch
test_batch_size = 10000; % Number of test examples processed in one go
learning_rate = 0.001; % Learning rate for gradient updates (step size)

% Tracking incorrect predictions during training and testing
training_errors = zeros(10, 1);  
test_errors_per_epoch = zeros(10, 1);  

for epoch = 1:10  % Train for 10 full passes over the dataset

    % Shuffle the dataset at the start of each epoch
    shuffle_idx = randperm(size(TRAIN_images, 1));
    TRAIN_images = TRAIN_images(shuffle_idx, :);
    TRAIN_answers = TRAIN_answers(shuffle_idx, :);
    TRAIN_labels = TRAIN_labels(shuffle_idx, :);

    disp("Starting a new epoch...");  

    % Mini-batch trainning
    for batch = 1:num_batches  
        % Extract mini-batch data
        start_idx = (batch - 1) * batch_size + 1;
        end_idx = start_idx + batch_size - 1;
        minibatch_images = TRAIN_images(start_idx:end_idx, :);
        minibatch_labels = TRAIN_labels(start_idx:end_idx, :);
    
        % Forward pass: Get the network's predictions
        net = zhang_forward_relog(net, minibatch_images');  
        
        % Compute softmax output (probabilities for each digit class)
        output_scores = net.v{4}; 
        softmax_output = exp(output_scores) ./ sum(exp(output_scores), 1);

        % Identify predicted labels (highest probability)
        [~, predicted_labels] = max(softmax_output);
        [~, true_labels] = max(minibatch_labels');  
        training_errors(epoch) = sum(predicted_labels ~= true_labels);  % Count misclassifications

        % Compute the loss and gradient for backpropagation
        error_signal = softmax_output - TRAIN_labels(start_idx:end_idx, :)';  
        loss = 0.5 * (error_signal * error_signal');  

        % Backward pass: Adjust weights using computed gradients
        net = zhang_backprop_relog(net, error_signal);  

        % Apply Adam optimizer to update weights and biases
        net = adam(net, net.dL_dW, net.dL_db, learning_rate, 0.9, 0.999);
    
    end  
    
    % Testing
    % Run the entire test set through the network in one big batch

    % Shuffle test data for fairness
    shuffle_idx = randperm(size(TEST_images, 1));
    TEST_images = TEST_images(shuffle_idx, :);
    TEST_answers = TEST_answers(shuffle_idx, :);
    TEST_labels = TEST_labels(shuffle_idx, :);
    
    % Extract full test set
    minibatch_images = TEST_images(1:test_batch_size, :);
    minibatch_labels = TEST_labels(1:test_batch_size, :);

    % Forward pass: Run test images through the trained network
    net = zhang_forward_relog(net, minibatch_images');  

    % Compute softmax output for test data
    output_scores = net.v{4}; % Use final layer pre-activations
    softmax_output = exp(output_scores) ./ sum(exp(output_scores), 1);

    % Identify incorrect test predictions
    [~, predicted_labels] = max(softmax_output);
    [~, true_labels] = max(minibatch_labels');
    test_errors_per_epoch(epoch) = sum(predicted_labels ~= true_labels); %Count test errors
    
end

% Display test error counts per epoch
disp([test_errors_per_epoch; "Test Errors Per Epoch"]);  