function [X, Y] = boundary(shape, filename)
    n = sum(shape);
    dim = length(shape);

    UR = rand(n, dim);
    LL = rand(n, dim);
    
    shape = [0, shape]
    for d = 1 : dim
        LL(1+shape(d-1):1+shape(d-1)+shape(d), d) = 0;
        % LL(shape(1)+1: n, 2) = 0;
    
        
        UR(1+shape(d-1):1+shape(d-1)+shape(d), d) = 1;
        % UR(shape(1)+1: n, 2) = 1;
    end
    X = [LL;UR];

end