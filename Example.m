clc
clear
addpath("GLQRC","LQRC","FixedLaplacianLQRC")

%% Define the problem

% Construct a Laplacian matrix for the example.
% The graph considered here is the one described in the paper.
D = 1;  % Depth of the tree
L = Example_ConstructLaplacian(D+1);
n = length(L(:,1));

% Number of integrators in the agents
M = 3;

% Matrices that define the cost
Z = eye(n*M);
R = eye(n) + diag(~rem(3:n+2,7))*99;
Q_block = eye(n);
Q = kron(diag([1,2,3]),Q_block);

% Define transformation matrices
S = diag(n:-1:1)-tril(ones(n,n));
S = S(:,1:end-1)./sqrt((n:-1:2).*(n-1:-1:1));
S_M = kron(eye(M),S);
Pi_M = kron(eye(M),S*S');

% Matrix that defines the sparsity constraits
ES = abs(L)<1e-8;

% Tolerance for the stopping criteria 
tolerance = 5e-4;

%% Define anonymous functions to compute the cost for a given gain K 
J = @(K) trace(lyap(S_M'*(kron(diag(ones(M-1,1),1),eye(n))-kron([zeros(M-1,1);1],eye(n))*K)*S_M,S_M'*Z*S_M)*S_M'*(Q+K'*R*K)*S_M);
J_compare = @(K1,K2) abs(J(K1)-J(K2))/min([J(K1),J(K2)])*100; 

%% Solve the GLQRC problem
fprintf(1,'\n:::------------      Solving the GLQRC problem      ------------:::\n')
K0 = kron([1;2;3]',L);
[K_glqrc,~] = glqrc(Z,Q,R,ES,K0,tolerance);
fprintf(1,'Cost: %6.3E\n',J(K_glqrc))

%% Solve the LQRC problem
fprintf(1,'\n:::------------      Solving the LQRC problem       ------------:::\n')
g0 = [1;2;3];
L0 = L;
[L_lqrc,g_lqrc,~] = lqrc(Z,Q,R,L0,g0,tolerance,false);
fprintf(1,'Cost: %6.3E\n',J(kron(g_lqrc',L_lqrc)))


%% Solve the LQRC problem for a symmetric case
fprintf(1,'\n:::----      Solving the LQRC problem (with symmetry)       ----:::\n')

% "Symmetrize" the graph (graph obtained by converting the directed edges in undirected ones)
L0 = L+L';
L0 = L0 - diag(diag(L0)) - diag(sum(L0-diag(diag(L0))));

g0 = [1;2;3];
[L_lqrc,g_lqrc,~] = lqrc(Z,Q,R,L0,g0,tolerance,true);
fprintf(1,'Cost: %6.3E\n',J(kron(g_lqrc',L_lqrc)))

%% Solve the LQRC problem using the proposed suboptimal approach
fprintf(1,'\n:::--- Solving the LQRC problem with the suboptimal approach ---:::\n')

% Step 1 - Determine the subptimal Laplacian
fprintf(1,':--  Step 1 - Determine the suboptimal Laplacian\n')
L0 = L;
L0 = sqrt(trace(lyap(-S'*L0'*S,S'*diag(1:n)*S))/trace(lyap(-S'*L0'*S,S'*L0'*R*L0*S)))*L0;   % Apply optimal scaling factor.
[L_sub,~] = glqrc(eye(n),Q_block,R,ES,L0,tolerance);

% Step 2 - Determine optimal coupling gains for the computed Laplacian
fprintf(1,':--  Step 2 - Determine optimal coupling gains for the computed Laplacian\n')
g0 = [1;2;6];
g_sub = fllqrc(L_sub,Z,Q,R,g0,tolerance);

% Compute the suboptimal cost
fprintf(1,'Cost: %6.3E\n',J(kron(g_sub',L_sub)))

%% Solve the GLQRC problem using the suboptimal approach to initialize
fprintf(1,'\n:::----  GLQRC problem initialized with suboptimal approach  ----:::\n')

% Step 1 - Determine the subptimal Laplacian
fprintf(1,':--  Step 1 - Determine the suboptimal Laplacian\n')
L0 = L;
L0 = sqrt(trace(lyap(-S'*L0'*S,S'*diag(1:n)*S))/trace(lyap(-S'*L0'*S,S'*L0'*R*L0*S)))*L0;   % Apply optimal scaling factor.
[L_sub,~] = glqrc(eye(n),Q_block,R,ES,L0,tolerance);

% Step 2 - Determine optimal coupling gains for the computed Laplacian
fprintf(1,':--  Step 2 - Determine optimal coupling gains for the computed Laplacian\n')
g0 = [1;2;6];
g_sub = fllqrc(L_sub,Z,Q,R,g0,tolerance);

% Step 3 - Build the gain and use it to initialize the GLQRC problem
fprintf(1,':--  Step 3 - Use the suboptimal gain to initialize the GLQRC problem\n')
[K_glqrc_subinit,~] = glqrc(Z,Q,R,ES,kron(g_sub',L_sub),tolerance);
fprintf(1,'Cost: %6.3E\n',J(K_glqrc_subinit))


%% Solve the LQRC problem using the suboptimal approach to initialize
fprintf(1,'\n:::----  LQRC problem initialized with Naive approach  ----:::\n')

% Step 1 - Determine the subptimal Laplacian
fprintf(1,':--  Step 1 - Determine the suboptimal Laplacian\n')
L0 = L;
L0 = sqrt(trace(lyap(-S'*L0'*S,S'*diag(1:n)*S))/trace(lyap(-S'*L0'*S,S'*L0'*R*L0*S)))*L0;   % Apply optimal scaling factor.
[L_sub,~] = glqrc(eye(n),Q_block,R,ES,L0,tolerance);

% Step 2 - Determine optimal coupling gains for the computed Laplacian
fprintf(1,':--  Step 2 - Determine optimal coupling gains for the computed Laplacian\n')
g0 = [1;2;6];
g_sub = fllqrc(L_sub,Z,Q,R,g0,tolerance);

% Step 3 - Build the gain and use it to initialize the LQRC problem
fprintf(1,':--  Step 3 - Use the suboptimal gain to initialize the LQRC problem\n')
[L_lqrc_init,g_lqrc_init,~] = lqrc(Z,Q,R,L_sub,g_sub,tolerance);
fprintf(1,'Cost: %6.3E\n',J(kron(g_lqrc_init',L_lqrc_init)))
