import numpy as np
import os
import argparse
import csv
from subprocess import Popen

def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not directory.endswith('/'):
        directory = directory + '/'
    return directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create batch synthetic data experiments.')
    parser.add_argument('experiment_name', help='The name of the experiment')
    parser.add_argument('--trials', type=int, default=30, help='The number of independent trials to run.')
    parser.add_argument('--scratch', default='/scratch/cluster/tansey/')
    
    parser.set_defaults()

    # Get the arguments from the command line
    args = parser.parse_args()

    exp_dir = make_directory(args.scratch, args.experiment_name)

    dims = (128, 128)
    signal_regions = [('big', (30, 30), (90, 90)), ('small', (50, 50), (80, 80))]
    signal_dists = [('well_separated', 2), ('flat_unimodal', 1)]
    signal_densities = [1., 0.5]
    noise_densities = [0., 0.05]

    gen_script = open(exp_dir + 'gendata.sh', 'wb')
    sfdr_script = open(exp_dir + 'sfdr_jobs', 'wb')
    hmrf_prep1_script = open(exp_dir + 'hmrf_prep1.sh', 'wb')
    hmrf_prep2_script = open(exp_dir + 'hmrf_prep2.sh', 'wb')
    hmrf_run_script = open(exp_dir + 'hmrf_run_jobs', 'wb')
    hmrf_post_script = open(exp_dir + 'hmrf_post.sh', 'wb')
    fdrl_script = open(exp_dir + 'fdrl_jobs.sh', 'wb')
    bh_script = open(exp_dir + 'bh_jobs.sh', 'wb')
    oracle_script = open(exp_dir + 'oracle_jobs.sh', 'wb')
    score_script = open(exp_dir + 'score.sh', 'wb')

    sfdr_script.write("""universe = vanilla
    Executable=/lusr/bin/python
    Requirements = InMastodon
    getenv = True
    +Group   = "GRAD"
    +Project = "AI_ROBOTICS"
    +ProjectDescription = "{0} sfdr benchmarks"
    """.format(args.experiment_name))

    sfdr_job = """Log = {0}sfdr_job.log
    Arguments = sfdr_run.py --data_file {0}data.csv --no_data_header --save_weights {0}sfdr_weights.csv --save_posteriors {0}sfdr_posteriors.csv --save_plateaus {0}sfdr_plateaus.csv --save_signal {0}sfdr_estimated_signal.csv --save_discoveries {0}sfdr_discoveries.csv --empirical_null --estimate_signal --solution_path --dual_solver graph graph --trails {0}trails.csv
    Output = {0}sfdr_job.out
    Error = {0}sfdr_job.error
    Queue 1
    """

    hmrf_run_script.write("""universe = vanilla
    Executable=run_hmrf.sh
    Requirements = InMastodon
    getenv = True
    +Group   = "GRAD"
    +Project = "AI_ROBOTICS"
    +ProjectDescription = "{0} hmrf benchmarks"
    """.format(args.experiment_name))

    hmrf_job = """Log = {0}hmrf_job.log
    Arguments = {0} {1}
    Output = {0}hmrf_job.out
    Error = {0}hmrf_job.error
    Queue 1
    """

    for region_name, region_start, region_end in signal_regions:
        for signal_dist, L in signal_dists:
            for signal_density in signal_densities:
                for noise_density in noise_densities:
                    subexp_dir = make_directory(exp_dir, '{0}_{1}_{2}_{3}'.format(region_name, signal_dist, signal_density, noise_density))
                    for trial in xrange(args.trials):
                        # Make the directories for the trial
                        trial_dir = make_directory(subexp_dir, str(trial))
                        plot_dir = make_directory(trial_dir, 'plots')

                        # Write the script that generates the data
                        gen_script.write('echo {0}\n'.format(trial_dir))
                        gen_script.write('gen2d {0}data.csv {0}true_weights.csv {0}true_signals.csv {0}oracle_posteriors.csv {0}edges.csv {0}trails.csv --width {1} --height {2} --region_min_x {3} --region_max_x {4} --region_min_y {5} --region_max_y {6} --region_weights {7} --default_weight {8} --signal_dist_name {9} --plot {10}data.pdf'.format(trial_dir, dims[0], dims[1], region_start[0], region_end[0], region_start[1], region_end[1], signal_density, noise_density, signal_dist, plot_dir))
                        gen_script.write('\n\n')

                        # Write the script that runs the smoothed fdr algorithm
                        sfdr_script.write(sfdr_job.format(trial_dir))
                        sfdr_script.write('\n\n')

                        # Write the script that preps the data for the HMRF benchmark
                        hmrf_prep1_script.write('matlab -r "hmrfPrep1 {0}";\n'.format(trial_dir))
                        hmrf_prep2_script.write('python hmrfPrep2.py {0}\n\n'.format(trial_dir))

                        # Write the script that runs the HMRF routine
                        hmrf_run_script.write(hmrf_job.format(trial_dir, L))
                        hmrf_run_script.write('\n\n')

                        # Write the script that processes the results of the HMRF routine and converts it to discoveries
                        hmrf_post_script.write('python hmrfPost.py {0}\n\n'.format(trial_dir))

                        # Write the script that runs the FDR-L routine
                        fdrl_script.write('python fdrl_run.py {0}\n\n'.format(trial_dir))

                        # Write the script that runs the Benjamini-Hochberg routine
                        bh_script.write('Rscript run_bh.r {0}data.csv {0}bh_discoveries.csv\n\n'.format(trial_dir))

                        # Write the script that processes the results of the oracle and converts it to discoveries
                        oracle_script.write('python oracle_run.py {0}\n\n'.format(trial_dir))

                        # Write the script that tallies the results and calculates TPR and FDR for each trial
                        score_script.write('python score.py {0} {1}scores.csv\n\n'.format(trial_dir, subexp_dir))

                    # Aggregate all the independent trials
                    score_script.write('python aggregate_scores.py {0}scores.csv {0}aggregate_scores.csv\n\n'.format(subexp_dir))

    gen_script.flush()
    gen_script.close()

    sfdr_script.flush()
    sfdr_script.close()

    hmrf_prep1_script.flush()
    hmrf_prep1_script.close()

    hmrf_prep2_script.flush()
    hmrf_prep2_script.close()

    hmrf_run_script.flush()
    hmrf_run_script.close()

    hmrf_post_script.flush()
    hmrf_post_script.close()

    fdrl_script.flush()
    fdrl_script.close()

    bh_script.flush()
    bh_script.close()

    oracle_script.flush()
    oracle_script.close()

    score_script.flush()
    score_script.close()


    # Create the surrogate script that condor uses to run the fdr smoothing algorithm on each job
    with open(exp_dir + 'sfdr_run.py', 'wb') as f:
        f.write('import sys; from smoothfdr import main; main()')

    test_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'test/')
    Popen('cp {0} {1}'.format(test_dir + 'hmrf*.py', exp_dir), shell=True)
    Popen('cp {0} {1}'.format(test_dir + '*.m', exp_dir), shell=True)
    Popen('cp {0} {1}'.format(test_dir + 'fdrl*.py', exp_dir), shell=True)
    Popen('cp {0} {1}'.format(test_dir + 'score.py', exp_dir), shell=True)
    Popen('cp {0} {1}'.format(test_dir + 'oracle_run.py', exp_dir), shell=True)
    Popen('cp {0} {1}'.format(test_dir + 'aggregate_scores.py', exp_dir), shell=True)
    #Popen('cp {0} {1}'.format(test_dir + '*.mexa64', exp_dir), shell=True)


    with open(exp_dir + 'README.txt', 'wb') as f:
        f.write("""Instructions for running benchmarks on Condor are as follows:
            -1) Make sure both pygfl and smoothfdr are properly installed and in your PATH. (local install: "pip install --user -e .")
            0) Go to the test directory, run matlab, and execute mex_setup()
            1) Run gendata.py [expname]
            2) Go to the [scratch]/[expname] directory
            3) Submit the sfdr jobs: condor_submit sfdr_jobs
            4) Make all the shell scripts runnable: chmod 777 *.sh
            5) Run the first hidden MRF prep script: ./hmrf_prep1.sh
            6) Run the second hidden MRF prep script: ./hmrf_prep2.sh
            7) Submit the hidden MRF jobs: condor_submit hmrf_run_jobs
            8) Run the hidden MRF post-processsing script: ./hmrf_post.sh
            9) Run the scoring script: ./score.sh
        """)
    

# python gen2d.py {0}data.csv {0}weights.csv {0}truth.csv {0}oracle_posteriors.csv {0}edges.csv --width {1} --height {2} --region_min_x {3} --region_max_x {4} --region_min_y {5} --region_max_y {6} --region_weights {7} --default_weight {8} --signal_dist_name {9} --plot {10}data.pdf

# trial_dir, dims[0], dims[1], region_start[0], region_end[0], region_start[1], region_end[1], signal_density, noise_density, signal_dist, plot_dir

#     # Plot results
#     parser.add_argument('--plot', help='Plot the resulting data and save to the specified file.')


# python __init__.py --data_file {0}data.csv --generate_data --signals_file {0}truth.csv --save_weights {0}weights.csv --save_posteriors {0}posteriors.csv --save_plateaus {0}plateaus.csv --save_signal {0}fdrs_discoveries.csv --save_oracle_posteriors {0}oracle_posteriors.csv --empirical_null --estimate_signal --solution_path 2d --width {1} --height
