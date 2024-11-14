
import os
import shutil
import utils


def setup_LifEnv():

    new_env_path = input('Type the desired file path for your processing environment and press enter.\n')
    dir_path = os.path.dirname(os.path.realpath(__file__))

    utils.generate_folder(new_env_path + '\\bin', use_local_dir=False)

    file_arr = ['config.txt', 'cts_file_headers.txt', 'cts_metadata.txt', 'html_plots_prefs.txt'
                , 'misalligned_files.txt']
    for filename in file_arr:
        shutil.copyfile(dir_path + '\\bin\\' + filename, os.path.join(new_env_path + '\\bin', filename))

    config_file = open(os.path.join(new_env_path + '\\bin\\config.txt'), 'a')

    config_file.write(new_env_path)

    config_file.close()

    shutil.copyfile(dir_path + '\\reprocessing_examp.py', os.path.join(new_env_path + '\\reprocessing_examp.py'))

    for name in ['bin_data', 'HK_data', 'processed_data']:
        utils.generate_folder(new_env_path + '\\data\\%s' % name, use_local_dir=False)

    for name in ['diagnostics']:
        utils.generate_folder(new_env_path + '\\figures\\%s' % name, use_local_dir=False)

    print('The LIF processing environment has been successfully setup!')


if __name__ == '__main__':
    setup_LifEnv()

