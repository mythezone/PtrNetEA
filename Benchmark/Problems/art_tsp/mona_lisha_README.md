Robert Bosch, February 2009
mona-lisa100K.tsp
$1000 Prize Offered
In February 2009, Robert Bosch created a 100,000-city instance of the traveling salesman problem (TSP) that provides a representation of Leonardo da Vinci's Mona Lisa as a continuous-line drawing. Techniques for developing such point sets have evolved over the past several years through work of Bosch and Craig Kaplan.

An optimal solution to the 100,000-city Mona Lisa instance would set a new world record for the TSP. If you have ideas for producing good tours or for showing that a solution is optimal, this is a very nice challenge problem! I would be happy to report any computational results you obtain on this example. The data set in TSPLIB format is given in the file

mona-lisa100K.tsp
The following papers describe various aspects of the mathematics behind the selection of city locations for the Mona Lisa TSP.

Robert Bosch, Opt Art, Math Horizons, February 2006, pages 6--9.
Craig Kaplan and Robert Bosch, TSP Art, Renaissance Banff: Bridges 2005: Mathematical Connections in Art, Music and Science, pages 301-308.
Robert Bosch and Adrianne Herman, Continuous Line Drawings via the Traveling Salesman Problem, Operations Research Letters, 2004, Volume 32, pages 302--303.
Ivars Peterson, Artful Routes, Mathematical Association of America Online, January 3, 2005.
Pretty examples of other TSP drawings can be found on Robert Bosch's TSP Art page and on the page of Craig Kaplan. Robert Bosch describes his work and the Mona Lisa TSP Challenge in the following video.

Robert Bosch interview


Status

The current best known results for the Mona Lisa TSP are:

Tour:  5,757,191     Bound:  5,757,084     Gap:  107 (0.0019%)

The tour was found on March 17, 2009, by Yuichi Nagata. The bound gives a value B such that no tour has length less than B; this bound was found on July 27, 2012, with the Concorde code. The Gap number is the gap in our knowledge, that is, the difference between the length of the tour and the value of the bound.
It has been over three years since Yuichi Nagata produced the best-known tour. To help perk up interest in searching for an even better solution, we are offering a $1,000 prize to the first person to find a tour shorter than 5,757,191.



Computation Log

July 27, 2012:  GAP=107.  A truncated branch-and-cut search, using an an artificial uppper bound of 5757092 and the LP from April 18, was terminated after 11.5 CPU years and 20,787 search nodes. The run improved the gap by 4 units.

April 18, 2012:  LP = 5757070.2.  The cut pool from the February 16, 2012 run allowed Concorde to push the LP bound up by three units. The LP solution vector is given in the file mona.6nsub.x and a drawing of the solution is given in mona.6nsub.pdf. To restart Concorde at this LP, use the master file mona-lisa100K.mas and the save file sav.6n.sub. I started a branch-and-cut search with this LP, aiming to lower the optimality gap to 99. That may be optimistic, but fortune favors the brave.

February 16, 2012:  GAP=111.  A truncated branch-and-cut search, using an an artificial uppper bound of 5757080 and the LP from January 21, completed in 5.6 CPU years and 5,073 search nodes. The run improved the gap by 8 units.

January 21, 2012:  LP = 5757067.2.  Repeated runs of Concorde's cutting-plane routines, equipped with a large pool of cuts obtained in the branch-and-cut search begun on July 7, 2011, increased the LP bound by 12 units over the previous best obtained on July 7, 2011. The increase is small, given the amount of computation involved, but it does amount to closing roughly 9% of the remaining gap between the LP bound and the best known tour. The LP solution vector is given in the file mona.5zsub.x and a drawing of the solution is given in mona.5zsub.pdf. We need better separation routines!

November 25, 2011:  GAP=119.  The lower bound improved by another 3 units to 5,757,072, with the ongoing branch-and-cut search. That is the good news. The bad news is that the cummulative CPU time is up to 13.9 years and the number of active subproblems is up to 957. The computation is clearly running out of steam.

September 12, 2011:  GAP=122.  Still chugging along with the branch-and-cut search. The lower bound has improved another 6 units to 5,757,069. The number of active subproblems is up to 267.

August 10, 2011:  GAP=128.  The truncated branch-and-cut search is now running on a network of 48 processor cores. The cummulative CPU time is up to 418 days and the number of active subproblems has grown to 52. The good news is that the lower bound has improved by another 4 units.

August 1, 2011:  GAP=132.  After 16 days of computing (on a single processor), a truncated branch-and-cut search starting from the July 7, 2011 LP relaxation has improved the lower bound by 3 units. The search tree currently has 7 active subproblems, using the artificial upper bound of 5757080. A drawing of the search tree is given here.

July 15, 2011:  Verified LP Bound.  Given the long computations to obtain the current LP bound, it made sense to spend the time to certify the accuracy of the LP relaxation. This took several days, mainly to run independent checks of all inequalities found by the local-cuts routines. Some details can be found here. The LP solution vector is given in the file mona.4zsub.x and a drawing of the solution is given in mona.4zx.pdf. To see the fractional values in the LP solution you can zoom in on the pdf file in areas where there are bits of color in the drawing.

July 7, 2011:  LP = 5757055.2, Gap = 135.  Some nice progress on the LP relaxation: the 137 gap used a search tree with over 18,000 subproblems, but the new 135 gap used a single LP only! Applying the huge pool of cutting planes found in the branching run brought the LP up to 5757041. The remaining improvement came from a tweak in Concorde's local-cuts code, allowing the code to search for local cuts in chunks (subproblems) of size up to 64. Previously the code was limited to chunks of subproblems of size 48. This, of course, suggests that we should go up to chunks of size 100 or so, but this will be tough to carry out efficiently.

May 11, 2011:  Gap = 137.  I let the branching run continue, although it was clear that not much progress would be made. The number of search nodes grew to 18,035, improving the lower bound to 5,757,054, before I terminated the computation. The total computing time for the search was 30.4 years.

March 29, 2011:  Gap = 138.  Couldn't stand the sight of several idle compute servers, so I continued the branching run that was terminated on January 16. Over the past week, the bound was improved by one unit to 5,757,053. The number of active subproblems has grown to 8,748.

January 22, 2011:  LP = 5757038.1.  Working with the massive set of 7.4 million cutting planes gathered in the previous branch-and-cut run, the LP relaxation of the problem was improved to a bound of 5757038.1. This is an improvement of 6.3 units over the previous LP relaxation.

January 16, 2011:  Gap = 139.  A truncated branch-and-cut run was started on September 18, 2010, with an artificial upper bound of 5,757,060. It was terminated after 120 days, accumulating 17.4 years of computer time by running in parallel on available workstations. The search tree grew to 6,236 active subproblems (and 14,753 nodes in the tree). The worst of the LP bounds for the subproblems is 5757051.38, yielding a new overall lower bound of 5,057,052. This is an improvement of 8 units over the previous best lower bound. The very small improvement, after so much CPU time, again indicates that new techniques will be needed to finally solve this example.

January 18, 2010:  LP = 5757031.8.  The cutting planes gathered in the November 5, 2009 run were used to improve the LP relaxation. The new LP bound is 5757031.8, an improvement of 3.5 units over the previous bound. This improvement is rather small, given that the gap to the best known tour is 159.2.

January 10, 2010:  GAP = 147.  A truncated branch-and-cut run was started on November 5, 2009. It was terminated after 66 days, accumulating 4.37 years of computer time by running in parallel on available workstations. The search tree grew to 533 active subproblems (and 1,065 nodes in the tree). The worst of the LP bounds for the subproblems is 5757043.24, yielding a new overall lower bound of 5057044. This improvement is considerable smaller than that obtained in the previous branch-and-cut run. Possibly the approach is running out of steam, suggesting new cutting-plane separation methods are needed.

November 5, 2009:  LP = 5757028.3, GAP = 162.  The cutting planes gathered during the April 16 truncated branch-and-cut run were used to further improve the linear-programming (LP) relaxation for the Mona Lisa TSP. The resulting LP value actually went above the bound from the branching tree, reaching an LP bound of 5757028.3. This is good news, since we can now start a new branch-and-cut run from this LP. The new LP was created by repeatedly applying the new cuts via a procedure that modifies them to better fit the current LP relaxation (a "tightening" procedure).

August 1, 2009:  GAP = 166.  The truncated branch-and-cut run was terminated on August 1 following a power outage at Georgia Tech. The run could be restarted from where it left off, but after 116 days it is a good stopping point. The search tree grew to 130 active subproblems (and 299 nodes in the tree). The worst of the LP bounds for the subproblems is 5057024.55. So the new overall lower bound is 5057025!

May 16, 2009:  GAP = 186.  The truncated branch-and-cut search that was started with the April 7 LP relaxation continues to run. After 40 days, the search has taken sixteen branching steps. The worst of the LP bounds for the seventeen subproblems is 5757004.76, yielding an overall lower bound of 5,757,005. The run is being carried out in parallel on 13 processor cores. A drawing of the search tree is given here. In the drawing, the height of a node gives the LP bound of the corresponding subproblem; the red nodes represent subproblems that have not yet been processed with additional cutting planes and the magenta nodes are ready to be split into further subproblems.

April 16, 2009:  GAP = 197.  A new truncated branch-and-cut search was started with the April 7 LP relaxation. An artificial upper bound of 5,757,057 was used this time. After 10 days, the search has taken two branching steps, creating a total of three subproblems. The LP bounds for the three problems are currently 5756994.17, 5756994.02, and 5756993.02. Since we know that all tours have integer length, an overall lower bound of 5,756,994 has been established. (This is the worst of the three LP bounds rounded up to the next integer.) The three subproblems continue to run, using three processor cores.

April 7, 2009:  LP = 5756981.2  A main goal of the March 30 truncated branch-and-cut run was to gather cutting planes that can be used to improve the linear-programming (LP) relaxation for the Mona Lisa TSP. Using the March 30 cutting planes and starting with the March 4 LP relaxation, Concorde improved the LP bound to 5,756,982. This is an improvement of 47 units over the March 4 LP and it is nearly the value of the bound obtained in the truncated branch-and-cut run. We can therefore expect a further improvement in the bound by again running a branch-and-cut search. The Concorde options used in the run were -mC48 -Z3. The log file can be found here.

March 30, 2009:  GAP = 206.  The lower bound was improved using a short branch-and-cut run with Concorde and the artificial upper bound of T = 5,756,985. Using the possibly invalid upper bound T allows Concorde to eliminate many variables from the LP relaxation, obtaining a problem that can be managed in a branch-and-cut search. ("Possibly invalid" here means that we do not actually know a tour of length T.) This type of run can establish a lower bound of up to T, but no higher. In this particular case, splitting the problem into two subproblems and applying Concorde established the bound of 5,765,985. This result gives a gap of 206 (0.0358%) to the tour of Yuichi Nagata. The Concorde run used 25.3 days of computing time on a 2.66 GHz Intel Xeon 5355 processor.

March 17, 2009:  TOUR = 5757191, GAP = 256.  Yuichi Nagata has improved the best known tour result by 8 units! His solution was found via series of computations with a genetic algorithm that he has designed. The new record tour has length 5,757,191 and it is given in the file: monalisa_5757191.tour.

March 16, 2009:  TOUR = 5757199, GAP = 264.  Keld Helsgaun improved his previous best tour by a single unit, using a stronger form of LKH's 10-opt moves. The new tour of length 5,757,199 is given in the file: monalisa_5757199.tour.

March 8, 2009:  TOUR = 5757200, GAP = 265.  Keld Helsgaun found another improvement, using 10-opt submoves on his March 7 tour. His new tour has length 5,757,200, improving the previous best known result by 3 units. The tour in TSPLIB format is given in the file: monalisa_5757200.tour.

March 7, 2009:  TOUR = 5757203, GAP = 268.  Keld Helsgaun improved the tour again with LKH, obtaining an improvement of 2 units from his previous tour (so 1 unit better than the tour obtained by Dong et al.). The new tour has length 5,757,203, giving a gap of 268 to the Concorde bound. Helsgaun's tour in TSPLIB format is given in the file: monalisa_5757203.tour.

March 6, 2009:  TOUR = 5757204, GAP = 269.  Changxing Dong, Christian Ernst, Gerold Jaeger, Dirk Richter and Paul Molitor used Helsgaun's (March 4) tour as a starting point and improved the tour by 1 unit. The new tour of length 5,757,204 is given in the file: monalisa_5757204.tour.

March 4, 2009:  LP = 5756934.4, GAP = 270.  Running Concorde, starting with the previous linear-programming relaxation, and now with the option -mC44 -Z3, improved the lower bound by 66 units to 5,756,935. The -Z3 flags turn on the use of domino-parity constraints. The gap to the best known tour is now 270 (0.00469%). The log file for the Concorde run can be found here.

March 2, 2009:  TOUR = 5757205, GAP = 336.  Keld Helsgaun found a new record tour, improving the previous best known result by 36 units! The new tour has length 5,757,205, giving a gap of only 336 units (0.00587%) to the Concorde bound from February 24. Helsgaun found his tour by using parallel LKH to obtain tours of length 5,757,264 and 5,757,269. The common edges in these two tours and the previous 5,757,245 tour were fixed, and the resulting smaller instance was run with high-order k-opt submoves. Helsgaun's tour in TSPLIB format is given in the file: monalisa_5757205.tour.

March 2, 2009:  TOUR = 5757241, GAP = 372.  Changxing Dong, Christian Ernst, Gerold Jaeger, Dirk Richter and Paul Molitor used a combination of Karp partitioning and LKH-2 to improve Helsgaun's February 26 tour by 4 units (using Helsgaun's tour as a starting point in their run). The tour of length 5,757,241 is 372 units (0.00646%) longer than the Concorde bound. The new tour in TSPLIB format is given in the file: monalisa_5757241.tour.

February 26, 2009:  TOUR = 5757245, GAP = 376.  Keld Helsgaun found a new best known tour using a parallel version of LKH. The tour of length 5,757,245 is only 376 units (0.00653%) longer than the Concorde bound established on February 24. Helsgaun's tour in TSPLIB format is given in the file: monalisa_5757245.tour.

February 26, 2009:  TOUR = 5757795, GAP = 926.  Keld Helsgaun reports that a 1,000-trial run of LKH-2 with 5-opt moves found a tour of length 5,758,239, while a single trial run with 8-opt moves found a tour of length 5,757,795. Log files for the two LKH-2 runs can be found here-5opt and here-8opt.

February 24, 2009:  LP = 5756868.8, GAP = 1962.  A second run of Concorde, starting with the linear-programming relaxation produced in the February 22 run, and now with the option -mC48, produced a lower bound of 5,756,869. This gives a gap of 1,962 (0.0341%) to the LKH tour. The log file for the Concorde run can be found here.

February 23, 2009:  TOUR = 5758831, GAP = 2212.  An initial LKH run found a tour of length 5,758,831 using 1,000 trials. The tour in TSPLIB format is given in the file: monalisa_5758831.tour. The log file for the LKH run can be found here.

February 22, 2009:  LP = 5756618.1.  Running Concorde with the -mC32 option produced a lower bound of 5,756,619, thus no tour can be shorter than this amount. The log file for the Concorde run can be found here.