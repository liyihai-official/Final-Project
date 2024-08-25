function M = loadFromBinary(dimension, filename, dtype)
    fid = fopen(filename, 'rb');
    shape = fread(fid, dimension, 'uint32');
    M = fread(fid, prod(shape), dtype);
    shape
    shape = fliplr(eye(dimension)) * shape;
    permu = dimension:-1:1;
    M = reshape(M, shape');
    M = permute(M, permu);
end