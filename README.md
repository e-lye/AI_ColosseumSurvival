# AI for Colosseum Survival! (A 2-Player Game)

In this project, Jack and I have developed an AI agent after exploring several results from our implementations of Alpha-Beta Pruning, Iterative-Deepening Search and Monte-Carlo Tree Search.

## Repository Description

* **ColosseumSurvival_Game** (Folder)
    * Contains the base repository provided to us, used to play the game and write our agents in.
* **OurAgents** (Folder)
    * Contains all the AI agents Jack and I have developed in our research. Our final and strongest agent being *student_agent.py*.
* **AI_AnalysisReport.pdf** (PDF)
    * Our written report of our implementation process as well as analysis of methods throughout our development.


## Process of Development
The detailed explanation of our progression can be read in *AI_AnalysisReport.pdf*. To summarize:
* We started off with a fairly simple look-ahead greedy algorithm that avoided moves that could lose in 1-2 moves, and took moves that could win immediately.
* We then built our first iteration of Alpha-Beta pruning that looked further ahead, that used a similar heuristic as our greedy algorithm, grouping our moves into Win Moves, Safe Moves, and Unsafe moves.
* Next,


## Authors

* Eamonn Lye [(Github)](https://github.com/e-lye)
* Jack Wei [(Github)](https://github.com/jyiwei)
