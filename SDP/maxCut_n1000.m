function [res,z,x_res] = maxCut_n1000(n, d, seed, num_rounding)
    n = int64(n);
    d = int64(d);
    W = py.regular_graph.regular_graph(n, d, seed);
    W = double(W);

    %startup
   
    [res,x_res]=maxcut_SDPNAL(W, num_rounding);
    n = double(n);
    d = double(d);
    z = (res/n - d/4)/sqrt(d/4);
    
    
    path_output = '...';
    res_name = sprintf('xxx.mat',n,d,seed);
    path_plus_resname = fullfile(path_output, res_name);
    save(path_plus_resname,'res','z','x_res');
end


