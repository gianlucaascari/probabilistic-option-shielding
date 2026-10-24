import logging
import os

def setup_logger(log_dir='./logs', log_name='city_simulation'):
    # Check if log directory exists, if not, create it
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filepath = os.path.join(log_dir, f"{log_name}.log")

    logging.basicConfig(filename=log_filepath,
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Add console logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    return logging.getLogger()