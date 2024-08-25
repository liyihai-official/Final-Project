function saveToBinary(M, dimension, filename, dtype)
    fid = fopen(filename, 'wb');

    shape = size(M);
    fwrite(fid, shape, 'uint32');

    permu = dimension:-1:1;
    M = permute(M, permu);
    fwrite(fid, M(:), dtype);
    fclose(fid);
end