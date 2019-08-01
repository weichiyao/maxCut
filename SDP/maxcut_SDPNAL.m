function [res,x_res]=maxcut_SDPNAL(W, num_rounding)
    d = sum(W,1);
    D = diag(d);
    L = D - W;
    n = size(W,1);
    
    model = ccp_model('maxcut');
        X = var_sdp(n,n);
        model.add_variable(X);
        model.maximize(inprod(L, X));
        model.add_affine_constraint(map_diag(X) == ones(n,1));
    model.solve;

    X=model.info.opt_solution{1};
    
    %% Rounding
    [u,s,v]=eig(X); 
    s=max(s,0);
    X = u*s*v'; 
    U = cholcov(X);
    res = zeros(1,num_rounding);
    x_res_all = zeros(num_rounding, n);
    for j = 1:num_rounding
        rr = size(U,1);
        x_res = sign(U'*randn(rr,1));
        res(j) = x_res'*L*x_res / 4;
        x_res_all(j,:) = x_res';
    end
    [res, idx] = max(res);
    x_res = x_res_all(idx,:);
end