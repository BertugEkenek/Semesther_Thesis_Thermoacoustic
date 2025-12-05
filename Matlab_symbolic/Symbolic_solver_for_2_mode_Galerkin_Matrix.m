clc; clear;
%% --- SYMBOLIC DEFINITIONS ---
syms mur11 mui11 mur12 mui12 mur21 mui21 mur22 mui22 real
syms C11 C12 C21 C22 Crn1 Cin1 Cr2n1 Ci2n1 real
syms c s real  % cos and sin as symbolic variables

%% --- MATRIX ELEMENTS ---
M11 = (Crn1 + 1i*Cin1)*(1 - (mur11 + 1i*mui11)*C11*(c - 1i*s)) - 1;
M12 = -(Crn1 + 1i*Cin1)*(mur12 + 1i*mui12)*C12*(c - 1i*s);
M21 = -(Crn1 + 1i*Cin1)*(mur21 + 1i*mui21)*C21*(c - 1i*s);
M22 = (Crn1 + 1i*Cin1)*(1 - (mur22 + 1i*mui22)*C22*(c - 1i*s)) - (Cr2n1 + 1i*Ci2n1);

%% --- MAIN DETERMINANT ---
det_expr = expand(M11*M22 - M12*M21);
Re_expr = simplify(real(det_expr));
Im_expr = simplify(imag(det_expr));

%% --- MU TERMS (12 total) ---
% Order: mur11, mui11, mur22, mui22, mur11*mur22, mur11*mui22, mui11*mur22, mui11*mui22, mur12*mur21, mur12*mui21, mui12*mur21, mui12*mui21
mu_all = [mur11, mui11, mur22, mui22, ...
          mur11*mur22, mur11*mui22, mui11*mur22, mui11*mui22, ...
          mur12*mur21, mur12*mui21, mui12*mur21, mui12*mui21];
mu_linear = [mur11, mui11, mur22, mui22, mur12, mui12, mur21, mui21];

%% --- EXTRACT COEFFICIENTS ---
n_terms = length(mu_all);
A = sym(zeros(2, n_terms));

% Process products first (higher order terms) - indices 5-12
Re_temp = Re_expr;
Im_temp = Im_expr;

for i = 5:n_terms  % Product terms
    vars_in_product = symvar(mu_all(i));
    
    % Set all other mu vars to 0
    Re_subs = Re_temp;
    Im_subs = Im_temp;
    for j = 1:length(mu_linear)
        if ~ismember(mu_linear(j), vars_in_product)
            Re_subs = subs(Re_subs, mu_linear(j), 0);
            Im_subs = subs(Im_subs, mu_linear(j), 0);
        end
    end
    
    % Take derivative with respect to each variable in product
    coeff_Re = Re_subs;
    coeff_Im = Im_subs;
    for v = vars_in_product
        coeff_Re = diff(coeff_Re, v);
        coeff_Im = diff(coeff_Im, v);
    end
    
    A(1, i) = simplify(coeff_Re);
    A(2, i) = simplify(coeff_Im);
    
    % Subtract this term from expressions
    Re_temp = Re_temp - A(1,i)*mu_all(i);
    Im_temp = Im_temp - A(2,i)*mu_all(i);
end

% Now process linear terms - indices 1-4
for i = 1:4  % Linear terms (only mur11, mui11, mur22, mui22)
    coeff_Re = diff(Re_temp, mu_all(i));
    coeff_Im = diff(Im_temp, mu_all(i));
    
    A(1, i) = simplify(coeff_Re);
    A(2, i) = simplify(coeff_Im);
    
    Re_temp = Re_temp - A(1,i)*mu_all(i);
    Im_temp = Im_temp - A(2,i)*mu_all(i);
end

% What remains is the constant term
b = [simplify(Re_temp); simplify(Im_temp)];

%% --- DISPLAY RESULTS ---
disp('========================================');
disp('SYMBOLIC RESULTS');
disp('========================================');
disp(' ');
disp('Matrix A (coefficients of mu terms):');
disp('Row 1: Real part, Row 2: Imaginary part');
disp('Columns: mur11, mui11, mur22, mui22,');
disp('         mur11*mur22, mur11*mui22, mui11*mur22, mui11*mui22,');
disp('         mur12*mur21, mur12*mui21, mui12*mur21, mui12*mui21');
disp(' ');
disp(A);

disp(' ');
disp('Vector b (constant terms):');
disp('[Re_constant; Im_constant]');
disp(b);

%% --- PYTHON/NUMPY OUTPUT ---
disp(' ');
disp('========================================');
disp('PYTHON/NUMPY FORMAT');
disp('========================================');
disp(' ');
disp('block_M11M22_M12M21 = np.array([');
disp('    [  # Real part coefficients');

% Print Real part (row 1)
for i = 1:n_terms
    coeff_str = char(A(1,i));
    if i < n_terms
        fprintf('        %s,\n', coeff_str);
    else
        fprintf('        %s\n', coeff_str);
    end
end

disp('    ],');
disp('    [  # Imaginary part coefficients');

% Print Imaginary part (row 2)
for i = 1:n_terms
    coeff_str = char(A(2,i));
    if i < n_terms
        fprintf('        %s,\n', coeff_str);
    else
        fprintf('        %s\n', coeff_str);
    end
end

disp('    ]');
disp('], dtype=float)');
disp(' ');
disp('b_vector = np.array([');
fprintf('    %s,  # Real constant\n', char(b(1)));
fprintf('    %s   # Imaginary constant\n', char(b(2)));
disp('])');
disp(' ');
disp('# Mu vector order:');
disp('# mu = [mur11, mui11, mur22, mui22, mur11*mur22, mur11*mui22, mui11*mur22, mui11*mui22, mur12*mur21, mur12*mui21, mui12*mur21, mui12*mui21]');