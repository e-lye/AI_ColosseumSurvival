# AI for Colosseum Survival! (A 2-Player Game)

In this project, Jack and I have developed an AI agent after exploring several results from our implementations of Alpha-Beta Pruning, Iterative-Deepening Search and Monte-Carlo Tree Search.

## Repository Description

* [**ColosseumSurvival_Game**](https://github.com/e-lye/AI_ColosseumSurvival/tree/main/ColosseumSurvival_Game)
    * Contains the base repository provided to us, used to play the game and write our agents in.
* [**OurAgents**](https://github.com/e-lye/AI_ColosseumSurvival/tree/main/OurAgents)
    * Contains all the AI agents Jack and I have developed in our research. Our final and strongest agent being *student_agent.py*.
* [**AI_AnalysisReport.pdf**](https://github.com/e-lye/AI_ColosseumSurvival/blob/main/AI_AnalysisReport.pdf) (PDF)
    * Our written report of our implementation process as well as analysis of methods throughout our development.


## Process of Development
The detailed explanation of our progression can be read in *AI_AnalysisReport.pdf*. To summarize:
* We started off with a fairly simple look-ahead greedy algorithm that avoided moves that could lose in 1-2 moves, and took moves that could win immediately.
* We then built our first iteration of an Alpha-Beta Pruning (ABP) agent that looked further ahead. We employed a similar heuristic to our greedy algorithm, grouping our possible moves into Win Moves, Safe Moves, and Unsafe Moves.
* Next, we built a fairly simple version of Monte-Carlo Tree Search, which is equally less thorough as it is expensive, which faired fairly well against our ABP. This soon became our benchmark for our agent's performance, and was used to judge the quality of our ABP agent as we continued to improve it. 
* Our next, final, and probably our biggest improvement was the use of Iterative-Deepening Search, which continued to increase the depth of search after each iteration. This was an ideal concept to use for our game, and probably many other competitive grid games, because of the following reason:
    * In the beginning, the list of moves that a player may take is very large, but at the same time, not as critical. IDS searches a shallow depth in this period, and can still make swift choices that remain safe since these moves are less condemning than they would be in the later part of the game.
    * As the game progresses, our list of available moves shrink but the choices we make become more critical. IDS allows us to search deeper and deeper depths due to our decrease in options, and thus allows us make more precisely correct choices over time.


## Developers

* Eamonn Lye [(Github)](https://github.com/e-lye)
* Jack Wei [(Github)](https://github.com/jyiwei)
