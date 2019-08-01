function [res, x_res] = SDP_max(W,num_rounding)
    [N, ~] = size(W);
    %% SDP
    %-- W is the adjacency matrix, of size N by N
    %-- N is the total number of the nodes
    cvx_begin sdp
        variable X(N,N) symmetric
        maximize trace(W*X) / 4
        X == semidefinite(N);
        diag(X) == ones(N,1);
    cvx_end
    %% Rounding
    U = chol(X);
    
    res = zeros(1,num_rounding);
    x_res_all = zeros(num_rounding, N);
    for j = 1:num_rounding
        x_res = sign(U'*randn(N,1));
        res(j) = x_res'*W*x_res / 4;
        x_res_all(j,:) = x_res';
    end
    [res, idx] = max(res);
    x_res = x_res_all(idx,:);
end