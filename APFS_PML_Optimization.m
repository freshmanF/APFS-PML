function [W, A, D, k_final] = APFS_PML_Optimization(X, Y_origin, HyperPara)

alpha = get_param(HyperPara, 'alpha', 1);
beta  = get_param(HyperPara, 'beta', 0.1);
lambda = get_param(HyperPara, 'lambda', 1);
k_initial = get_param(HyperPara, 'k_initial', 20);
eta   = get_param(HyperPara, 'eta', 0.1);
maxIter = get_param(HyperPara, 'maxIter', 100);
minLossMargin = get_param(HyperPara, 'minLossMargin', 1e-5);
step_size_D = get_param(HyperPara, 'step_size_D', 1e-4);
step_size_C = get_param(HyperPara, 'step_size_C', 1e-4);
step_size_V = get_param(HyperPara, 'step_size_V', 1e-4);
omega = get_param(HyperPara, 'omega', 0.01);
prune_threshold = get_param(HyperPara, 'prune_threshold', 1e-4);


k = k_initial;
[num_sample, num_feature] = size(X);
A = eye(num_sample);

[P, C, D, W, V] = initialize_model(X, Y_origin, k, num_feature);

old_loss = Inf;

for iter = 1:maxIter
    w_row_norms = sqrt(sum(W.^2, 2)) + eps; 
    B = diag(1 ./ (2 * w_row_norms));
    
    if k > 0
        A_W = C' * P' * P * C + 2 * alpha * B;
        B_W = C' * P' * D;
        if rcond(A_W) < 1e-6, W = pinv(A_W) * B_W; else, W = A_W \ B_W; end
        
        grad_V = -beta * P' * (D - P*V);
        V_temp = V - step_size_V * grad_V;
        V = SimplexProj(V_temp);

        grad_D = (D - P*C*W) + beta * (D - P*V);
        D_temp = D - step_size_D * grad_D;
        D_temp(Y_origin == 0) = 0; 
        D = SimplexProj_Matrix(D_temp);

        W_c_for_grad = diag(w_row_norms);
        grad_C = P' * (P*C*W - D) * W' + lambda * (P' * (P*C - X)) * W_c_for_grad;
        C = C - step_size_C * grad_C;
    end
    
   
    if k > 0
        
        Dist_sq_X = pdist2_weighted(X, C, w_row_norms'); 
        Dist_sq_V = pdist2(D, V, 'squaredeuclidean');
        Cost = lambda * Dist_sq_X + beta * Dist_sq_V;
        
        p_col_norms = sqrt(sum(P.^2, 1)) + eps;
        d_P = 0.5 ./ p_col_norms;
        
        Total_Cost = Cost + 2 * omega * repmat(d_P, num_sample, 1);
        
        P_exp = exp(-Total_Cost / (2 * eta));
        P = P_exp ./ sum(P_exp, 2);
        P(isnan(P)) = 1/k;
    end

    if k > 0
        col_norms_after_update = sqrt(sum(P.^2, 1));
        active_indices = find(col_norms_after_update > prune_threshold);
        
        if length(active_indices) < k
            new_k = length(active_indices);
            fprintf('Pruning prototypes: k changes from %d to %d\n', k, new_k);
            k = new_k;
            
            P = P(:, active_indices);
            C = C(active_indices, :);
            V = V(active_indices, :);
            
            P = P ./ (sum(P, 2) + eps);
        end
    end
    
    if k == 0, break; end

    Dist_sq_X_for_loss = pdist2_weighted(X, C, w_row_norms');
    loss_fit = 0.5 * norm(P*C*W - D, 'fro')^2;
    loss_W = alpha * sum(w_row_norms);
    loss_V_consistency = (beta/2) * norm(D - P*V, 'fro')^2;
    loss_P_cluster = (lambda/2) * sum(P .* Dist_sq_X_for_loss, 'all');
    loss_P_adap = omega * sum(sqrt(sum(P.^2, 1)));
    
    current_loss = loss_fit + loss_W + loss_V_consistency + loss_P_cluster + loss_P_adap;

    fprintf('Iter %d, k: %d, Loss: %.4f (P_adap: %.4f)\n', iter, k, current_loss, loss_P_adap);

    if iter > 5 && abs(old_loss - current_loss) < minLossMargin * abs(old_loss), break; end
    old_loss = current_loss;
end

k_final = k;
if k == 0
    W = zeros(num_feature, size(D, 2)); 
end
fprintf('--- Optimization Finished. Final k = %d ---\n', k_final);
end


function [P, C, D, W, V] = initialize_model(X, Y_origin, k, num_feature)
    [num_sample, ~] = size(X);
    num_label = size(Y_origin, 2);
    
    fprintf('Running initial K-means for C and P...\n');
    try
        opts = statset('Display','off', 'MaxIter',100);
        [G_idx, C] = kmeans(X, k, 'Replicates',5, 'Options',opts);
    catch
        G_idx = randi(k, num_sample, 1);
        C = X(randperm(num_sample, k), :);
    end
    P = full(sparse(1:num_sample, G_idx, 1, num_sample, k));

    D = SimplexProj_Matrix(Y_origin);
    V = zeros(k, num_label);
    if k > 0
        if rcond(P'*P) < 1e-6, V = pinv(P'*P) * (P'*D); else, V = (P'*P) \ (P'*D); end
        V = SimplexProj(V);
    end

    W = zeros(num_feature, num_label);
    if k > 0
        if rcond(C'*P'*P*C) < 1e-6, W = pinv(C'*P'*P*C + 0.1*eye(num_feature)) * (C'*P'*D);
        else, W = (C'*P'*P*C) \ (C'*P'*D); end
    end
end

function value = get_param(param_struct, field_name, default_value)
    if isfield(param_struct, field_name) && ~isempty(param_struct.(field_name))
        value = param_struct.(field_name);
    else
        value = default_value;
    end
end


function D2 = pdist2_weighted(X, Y, w_row_vec)
    if isempty(Y), D2 = zeros(size(X,1), 0); return; end
    X_weighted = X .* sqrt(w_row_vec); 
    Y_weighted = Y .* sqrt(w_row_vec);
    D2 = pdist2(X_weighted, Y_weighted, 'squaredeuclidean');
end

function Proj_M = SimplexProj_Matrix(M)
    [N, Q] = size(M); Proj_M = zeros(N, Q);
    for i = 1:N
        Proj_M(i, :) = SimplexProj_Row(M(i,:));
    end
end

function V_proj = SimplexProj(V)
    [k, q] = size(V); V_proj = zeros(k,q);
    if k == 0, return; end
    for i=1:k
        V_proj(i,:) = SimplexProj_Row(V(i,:));
    end
end

function p_proj_row = SimplexProj_Row(p_row)
    v_sorted = sort(p_row, 'descend'); Q = length(p_row);
    rho_vals = (cumsum(v_sorted) - 1) ./ (1:Q);
    rho_idx = find(v_sorted > rho_vals, 1, 'last');
    if isempty(rho_idx), theta = (sum(p_row) - 1) / Q; else, theta = rho_vals(rho_idx); end
    p_proj_row = max(p_row - theta, 0);
end