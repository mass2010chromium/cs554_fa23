import matplotlib.pyplot as plt

datasets = {}

with open("log") as infile:
    lines = infile.readlines()

for i in range(0, len(lines), 9):
    N = int(lines[i].split('=')[1])
    serial_time = int(lines[i+2].split(' ')[0])
    serial_time += int(lines[i+4].split(' ')[0])
    parallel_time = int(lines[i+6].split(' ')[0])
    parallel_time += int(lines[i+8].split(' ')[0])
    parallel_n = int(lines[i+5].split('(')[1].split(' ')[0])

    if parallel_n not in datasets:
        datasets[parallel_n] = [[], [], [], []]
    Ns, serial_times, parallel_times, speedup = datasets[parallel_n]
    Ns.append(N)
    serial_times.append(serial_time)
    parallel_times.append(parallel_time)
    speedup.append(serial_time / parallel_time)

plt.figure()
for k, [N, serial, parallel, speedup] in datasets.items():
    plt.plot(N, speedup, label=f"num_threads={k}")
    plt.legend()
plt.axhline(y=1, color='r', linestyle='--')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Relative speedup')
plt.title("Speedup versus problem size for differing numbers of threads")
plt.show()

