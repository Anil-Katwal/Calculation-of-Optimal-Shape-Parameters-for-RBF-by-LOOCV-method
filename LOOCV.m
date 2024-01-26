% Define the three-dimensional "bumpy" function f(x, y, z)
f = @(x, y, z) 64 * x .* (1 - x) .* y .* (1 - y) .* z .* (1 - z);
k = 600; 
n_points = k;
p = haltonset(3,'Skip',1e3,'Leap',1e2);
halton_points = net(p, n_points);
DM = pdist2(halton_points, halton_points);
% Initialize variables
min_ep = 0;
max_ep = 4; 
rhs = f(halton_points(:,1), halton_points(:,2), halton_points(:,3));
% Define the MQ RBF function
rbf_mq = @(ep, r) sqrt((ep*r).^2 + 1);
cost_function = @(ep) CostEps(ep, rbf_mq, DM, rhs);
[optimal_ep, fval] = fminbnd(cost_function, min_ep, max_ep);
disp(['Optimal Shape Parameter (epsilon): ', num2str(optimal_ep)]);
x = halton_points(:,1);
y = halton_points(:,2);
z = halton_points(:,3);
Z = f(x, y, z);
A = rbf_mq(optimal_ep, DM);
% Add a small constant to the diagonal
b = A \ Z;
% Generate a grid for reconstruction
[M, N, O] = meshgrid(0:0.025:1);
x1 = M(:);
y1 = N(:);
z1 = O(:);
DM1 = sqrt((x1 - x').^2 + (y1 - y').^2 + (z1 - z').^2);
B = rbf_mq(optimal_ep, DM1) * b;
E = reshape(B, size(M));
% Generate 300 additional Halton points for testing point
test_points = net(haltonset(3), 300);
x_test = test_points(:, 1);
y_test = test_points(:, 2);
z_test = test_points(:, 3);
Z_test = f(x_test, y_test, z_test);
DM_test = sqrt((x_test - x').^2 + (y_test - y').^2 + (z_test - z').^2);
A_test = rbf_mq(optimal_ep, DM_test);

b_test = A_test \ Z_test;
DM1_test = sqrt((x_test - x').^2 + (y_test - y').^2 + (z_test - z').^2);
B_test = rbf_mq(optimal_ep, DM1_test) * b_test;
exact_test = f(x_test, y_test, z_test);
abs_error = abs(B_test - exact_test);
max_abs_error = max(abs_error);
disp(['Maximum Absolute Error: ', num2str(max_abs_error)]);

% Interpolate exact_test onto a grid
[M_test, N_test, O_test] = meshgrid(0:0.025:1);
exact_test_interp = griddata(x_test, y_test, z_test, exact_test, M_test, N_test, O_test);

% Plot for exact function 
figure;
subplot(1,2,1);
slice(M_test, N_test, O_test, exact_test_interp, 0.5, 0.5, [0.25, 0.65]);
title('Slicing for exact function');
xlabel('x');
ylabel('y');
zlabel('z');

% subplot for absolute error
subplot(1,2,2);
[M_error, N_error, O_error] = meshgrid(0:0.025:1);
abs_error_interp = griddata(x_test, y_test, z_test, abs_error, M_error, N_error, O_error);
slice(M_error, N_error, O_error, abs_error_interp,0.5, 0.5, [0.25, 0.65]);
title('Slicing for absolute error function');
xlabel('x');
ylabel('y');
zlabel('z');

% Extract grid points and values for isosurface
iso_value_1 = 0.01;
iso_value_2 = 0.8;
iso_surface_1 = isosurface(M, N, O, E, iso_value_1);
iso_surface_2 = isosurface(M, N, O, E, iso_value_2);

figure;
title('Isosurfaces');
h1 = patch(iso_surface_1,'FaceColor','green','EdgeColor','none');
hold on;
h2 = patch(iso_surface_2,'FaceColor','red','EdgeColor','none');
view(3);
camlight;
lighting gouraud;
alpha(h1, 0.35);
alpha(h2, 0.35);
xlabel('X');
ylabel('Y');
zlabel('Z');
h_legend = legend([h1, h2], {'Iso Surface 1 at 0.01', 'Iso Surface 2 at 0.8'});

% Cost function for epsilon using LOOCV paper;
function ceps = CostEps(ep, rbf, DM, rhs)
    A = rbf(ep, DM);
    invA = pinv(A);
    errorvector = (invA * rhs) ./ diag(invA);
    ceps = norm(errorvector);
end