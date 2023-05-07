import subprocess
import time
import datetime as dt

with open('ram_usage.txt', 'w') as file:
    file.write('time,mem_used,mem_avail\n')

while True:
    proc = subprocess.run(['free'], stdout=subprocess.PIPE)

    stdout = proc.stdout.decode('utf-8')

    line = stdout.split('\n')[1]
    line = line.split(' ')

    new_line = []
    for item in line:
        if item != '':
            new_line.append(item)

    line = new_line

    mem_used = int(line[2])
    mem_avail = int(line[6])

    now = str(dt.datetime.now())
    # print(now)
    # now_as_dt = dt.datetime.strptime(now, '%Y-%m-%d %H:%M:%S.%f')
    # print(now_as_dt)

    with open('ram_usage.txt', 'a') as file:
        file.write("%s,%d,%d\n" % (dt.datetime.now(), mem_used, mem_avail))

    time.sleep(30)
    