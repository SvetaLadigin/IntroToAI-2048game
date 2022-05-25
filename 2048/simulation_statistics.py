import subprocess

import numpy as np
import matplotlib
import os, sys
import subprocess
import pandas as pd

def parse_output(output):
    dict = {}
    item2 = output.splitlines()
    depth_list = []
    step_counter = 0
    # item2[-2]
    print(item2)
    for line in item2:
        if b'depth' in line:
            new_line = line.decode().split(" ")
            output_depth = new_line[-1]
            depth_list.append(int(output_depth))
            step_counter += 1
        if b'value' in line:
            new_line = line.decode().split(" ")
            output_tile = new_line[6]
            output_score = new_line[-1]
            dict['score'] = output_score
            dict['tile'] = output_tile
    avg = sum(depth_list)/step_counter
    dict['depth'] = avg
    return dict


list_of_outputs = []

f = open("statistics2.txt", "w")

for line in range(1):
    p = subprocess.Popen(["python3 -player1 ExpectimaxMovePlayer -player2 MiniMaxIndexPlayer -move_time 1"],
                         bufsize=2048, shell=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    p.wait()
    output = p.stdout.read()
    print(output)
    parsed_output = parse_output(output)
    final_append = [1, 'abmove', parsed_output.get('tile'), parsed_output.get('score'), parsed_output.get('depth')]
    list_of_outputs.append(final_append)
    print(list_of_outputs)
    for element in final_append:
        f.write(str(element) + " ")
    f.write("\n")
    p = subprocess.Popen(["python3 main.py -player1 ABMovePlayer -player2 MiniMaxIndexPlayer -move_time 2"],
                         bufsize=2048, shell=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    p.wait()
    output = p.stdout.read()
    print(output)
    parsed_output = parse_output(output)
    final_append = [2, 'abmove', parsed_output.get('tile'), parsed_output.get('score'), parsed_output.get('depth')]
    list_of_outputs.append(final_append)
    print(list_of_outputs)
    for element in final_append:
        f.write(str(element) + " ")
    f.write("\n")
    p = subprocess.Popen(["python3 main.py -player1 ABMovePlayer -player2 MiniMaxIndexPlayer -move_time 4"],
                         bufsize=2048, shell=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    p.wait()
    output = p.stdout.read()
    print(output)
    parsed_output = parse_output(output)
    final_append = [4, 'abmove', parsed_output.get('tile'), parsed_output.get('score'), parsed_output.get('depth')]
    list_of_outputs.append(final_append)
    print(list_of_outputs)
    for element in final_append:
        f.write(str(element) + " ")
    f.write("\n")
    p = subprocess.Popen(["python3 main.py -player1 MiniMaxMovePlayer -player2 MiniMaxIndexPlayer -move_time 1"], bufsize=2048, shell=True,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    p.wait()
    output = p.stdout.read()
    print(output)
    parsed_output = parse_output(output)
    final_append = [1, 'minimax', parsed_output.get('tile'), parsed_output.get('score'), parsed_output.get('depth')]
    list_of_outputs.append(final_append)
    print(list_of_outputs)
    for element in final_append:
        f.write(str(element) + " ")
    f.write("\n")
    p = subprocess.Popen(["python3 main.py -player1 MiniMaxMovePlayer -player2 MiniMaxIndexPlayer -move_time 2"], bufsize=2048, shell=True,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    p.wait()
    output = p.stdout.read()
    print(output)
    parsed_output = parse_output(output)
    final_append = [2, 'minimax', parsed_output.get('tile'), parsed_output.get('score'), parsed_output.get('depth')]
    list_of_outputs.append(final_append)
    print(list_of_outputs)
    for element in final_append:
        f.write(str(element) + " ")
    f.write("\n")
    p = subprocess.Popen(["python3 main.py -player1 MiniMaxMovePlayer -player2 MiniMaxIndexPlayer -move_time 4"], bufsize=2048, shell=True,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    p.wait()
    output = p.stdout.read()
    print(output)
    parsed_output = parse_output(output)
    final_append = [4, 'minimax', parsed_output.get('tile'), parsed_output.get('score'), parsed_output.get('depth')]
    list_of_outputs.append(final_append)
    print(list_of_outputs)
    for element in final_append:
        f.write(str(element) + " ")
    f.write("\n")

#
# data = {}
# for line in f:
#     params = line.split()
#     data['time'], data['agent'], data['tile'], data['score'], data['depth'] = params
# pd.DataFrame.from_dict(data)
# # print(list_of_outputs)
# # textfile = open("a_file.txt", "w")
# # for element in list_of_outputs:
# #     textfile.write(element + "\n")
# # textfile.close()
# #
# # f.close()