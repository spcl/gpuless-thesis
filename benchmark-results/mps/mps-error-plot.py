import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

sns.set_style("whitegrid")

def parse_out_file(file_path_list):
    res = []
    for file_path in file_path_list:
        x = []  # List to store x-axis values
        y = []  # List to store y-axis values
        with open(file_path, 'r') as file:
            lines = file.readlines()
            idx = 1
            for line in lines:
                # Splitting the line assuming it contains comma-separated values
                values = line.strip().split(' ')
                if "seconds" in values:
                    x.append(idx)  # Assuming first value is for x-axis
                    y.append(float(values[-2]))  # Assuming second value is for y-axis
                    idx += 1
            res.append(sum(y))
    return res

def plot_series_graph(x, y):
    data = []
    
    for i in range(len(x)):
        temp = []
        for j in range(3):
            temp.append(y[j + 3 * i])
        data.append(temp)

    plt.figure(figsize=(5, 3))
    
    num_entries = len(x)
    positions = np.arange(num_entries) * 4
    num_nested_plots = 3
    for i in range(num_entries):

        pos = positions[i] - 1 + np.array([0, 1, 2])  # Set positions for each nested group of boxplots
        print(pos)
       
        plt.boxplot(data[i][0], positions=[pos[0]], widths=3 / num_nested_plots, patch_artist=True)
        plt.text(pos[0], np.max(data[i][0]) + 0.5, f'No errors', ha='center', va='center')
        
        plt.boxplot(data[i][1], positions=[pos[1]], widths=3 / num_nested_plots, patch_artist=True)
        plt.text(pos[1], np.max(data[i][1]) + 0.5, f'4 errors', ha='center', va='center')
        
        plt.boxplot(data[i][2], positions=[pos[2]], widths=3 / num_nested_plots, patch_artist=True)
        plt.text(pos[2], np.max(data[i][2]) + 0.5, f'8 errors', ha='center', va='center')
    # Show plot
    plt.grid(True)
    plt.title('STREAM benchmark, 1000 invocations')  # Setting the title
    #plt.xlabel('Number of concurrent clients')
    plt.ylabel('Time [s]')  # Setting y-axis label
    print(positions)
    plt.xticks(positions, ['2 clients', '3 clients'])
    plt.ylim([13, 25.5])
    plt.xlim([-2, 6])
    #plt.show()
    # Save the plot as a PDF file
    plt.savefig('bar_graph_stream_boxplot.pdf', bbox_inches='tight',pad_inches = 0, transparent=False)

    #plt.show()

y_values = []

########################## 2 clients data ##########################
file_path = []
for i in range(1, 11):
    file_path.append(f'bench-results/test11/client2/test11-mps-bench-result-client1-{i}.out')
y_values.append(parse_out_file(file_path))

file_path = []
for i in range(1, 11):
    file_path.append(f'bench-results/test12/client2/test12-mps-bench-result-client1-{i}.out')
y_values.append(parse_out_file(file_path))

file_path = []
for i in range(1, 11):
    file_path.append(f'bench-results/test13/client2/test13-mps-bench-result-client1-{i}.out')
y_values.append(parse_out_file(file_path))

########################## 3 clients data ##########################
file_path = []
for i in range(1, 11):
    file_path.append(f'bench-results/test11/test11-mps-bench-result-client1-{i}.out')
y_values.append(parse_out_file(file_path))

file_path = []
for i in range(1, 11):
    file_path.append(f'bench-results/test12/test12-mps-bench-result-client1-{i}.out')
y_values.append(parse_out_file(file_path))

file_path = []
for i in range(1, 11):
    file_path.append(f'bench-results/test13/test13-mps-bench-result-client1-{i}.out')
y_values.append(parse_out_file(file_path))

plot_series_graph(['2 clients', '3 clients'], y_values)
