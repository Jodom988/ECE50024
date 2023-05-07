import subprocess
import time
import datetime as dt

def main():

    python_file = 'cond_gan_autoencoder_celeba.py'

    # sub = subprocess.run(['python3', python_file])

    while True:
        sub = subprocess.run(['python3', python_file, '-l'])
        
        print("=====================================")
        print("At time: ", dt.datetime.now())
        print("Subprocess returned: ", sub.returncode)
        print("=====================================")
        time.sleep(1)
    

if __name__ == '__main__':
    main()