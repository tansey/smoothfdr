function []=hmrfRun(expdir, L)
    % L = str2num(Lstr);
    input_info = [3, 130, 130, 0.1];
    reglocafile = strcat(expdir, 'bufferregions.csv');
    pvaluefile = strcat(expdir, 'bufferpvalues.csv');
    signalfile = strcat(expdir, 'buffer_signals_bh_hmrf.csv');
    zvaluefile = strcat(expdir, 'bufferdata.csv');
    output_estimate = strcat(expdir, 'buffer_estimate_result.csv');
    output_LIS = strcat(expdir, 'buffer_lis.csv');
    output_process = strcat(expdir, 'buffer_process_output.csv');
    

    result=fdrBH95(input_info,reglocafile,pvaluefile,signalfile);
    initials = initialvalue(signalfile, zvaluefile, reglocafile, L);

    input_info = [3, 130, 130, L, 1000, 5000, 1e6, 5000];
    computelis(input_info, initials, zvaluefile, reglocafile, output_estimate, output_LIS, output_process);
end
