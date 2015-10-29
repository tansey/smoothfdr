function []=hmrfPrep1(expdir)
    data = csvread(strcat(expdir,'data.csv'), 0, 0);
    flatdata = reshape(data, 1, []);
    dlmwrite(strcat(expdir,'flatdata.csv'), flatdata);
    flatdata_pvalues = normcdf(-abs(flatdata),0,1) * 2;
    dlmwrite(strcat(expdir,'flatdata_pvalues.csv'), flatdata_pvalues);
    exit
end