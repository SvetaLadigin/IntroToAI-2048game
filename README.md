# IntroToAI-2048game hw2 in Technion course intro to AI spring21

the 2048 game with a few game strategies 

2 players, move and index. 
Move for achieving hieghest score
Index for sabotaging the Move

move_players = 
  'GreedyMovePlayer'
 'ImprovedGreedyMovePlayer'
 'MiniMaxMovePlayer'
 'ABMovePlayer'
 'ExpectimaxMovePlayer'
 'ContestMovePlayer'
 
index_players = 
  'RandomIndexPlayer'
 'MiniMaxIndexPlayer'
 'ExpectimaxIndexPlayer'

run the game without players:
$ git branch -M main

with players:
$ python main.py -player1 [move name] -player2 [index name] -move_time [time for making a move]

example:
$ python main.py -player1 MiniMaxMovePlayer -player2 MiniMaxIndexPlayer -move_time 5

simulation statistics : for running multiple games and get score statistics

$ python simulation_statistics.py

graph simulation : creating a graph from the statistics 

$ python grapth_sim.py



