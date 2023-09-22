
def read_log(i):
    fn = f"{i}.out"
    with open(fn) as data_file:
        data = [float(x.strip()) for x in data_file.readlines()]

    return (data[0], data[1:])

if __name__ == "__main__":
    i = 0
    log_data = []
    while True:
        try:
            log = read_log(i)
            log_data.append(log)
        except Exception as e:
            #print(e)
            break
        i = i + 1

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data_all = []
    timings = []
    for timing, dat in log_data:
        timings.append(timing)
        data_all.extend(dat)

    timings = np.array(timings)
    print("Run time stats:")
    print("  Average:", np.mean(timings))
    print("      Max:", np.max(timings))
    print("      Min:", np.min(timings))
    print("      Std:", np.std(timings))
    plt.figure(0)
    plt.plot(data_all)
    plt.savefig("plot.png")
    plt.close()
        
