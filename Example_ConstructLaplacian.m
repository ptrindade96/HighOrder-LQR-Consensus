function [L] = Example_ConstructLaplacian(m)
    L_SC = [2 0 0 0 0 -1 -1;-1 2 0 -1 0 0 0;0 -1 2 0 0 0 -1;0 0 -1 2 0 -1 0;0 0 0 -1 2 0 -1;0 -1 0 0 -1 2 0;0 -1 0 -1 0 -1 3];
    n_SC = length(L_SC(1,:));
    nn = (1-2^m)/(1-2)*n_SC;    % Number of nodes for given m and n_SC

    I = eye(nn);
    e = @(i,n) I(1:n,i);
    M_L = -(e(1,n_SC)*e(6,n_SC)' + e(2,n_SC)*e(5,n_SC)');
    M_R = -(e(1,n_SC)*e(5,n_SC)' + e(2,n_SC)*e(4,n_SC)');
    S = e(1,n_SC)*e(1,n_SC)' + e(2,n_SC)*e(2,n_SC)';
    
    n = n_SC;
    L = L_SC;
    for i=1:m-1
        L(1:n_SC,1:n_SC) = L(1:n_SC,1:n_SC) + S;
        L = [L_SC,zeros(n_SC,2*n);[M_R;zeros(n-n_SC,n_SC)],L,zeros(size(L));[M_L;zeros(n-n_SC,n_SC)],zeros(size(L)),L];
        n = length(L(1,:));
    end
end

