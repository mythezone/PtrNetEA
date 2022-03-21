FHCP Challenge Set

The FHCP Challenge Set is a collection of 1001 instances of the Hamiltonian Cycle Problem, ranging in size from 66 vertices up to 9528 vertices, with an average size of just over 3000 vertices. The problems are in the standard HCP format (see below). All of the instances are designed to be difficult to solve using standard HCP heuristics. Every one of the 1001 graphs is known to be Hamiltonian.

If you use the FHCP Challenge Set in your research, we kindly request that you cite the following paper:

Haythorpe, M. “FHCP Challenge Set: The first set of structurally difficult instances of the Hamiltonian cycle problem.” Bulletin of the ICA, 83:98-107, 2018. PDF available

Download the FHCP Challenge Set

Download solutions for the FHCP Challenge Set

Short URL: http://fhcp.edu.au/fhcpcs

COMPETITION OPENS: 30TH SEPTEMBER, 2015
COMPETITION DEADLINE : 30TH SEPTEMBER, 2016

The FHCP team is offering a $1001 (US) prize for the first researcher, or team of researchers, who can provide a solution for every graph in the set. If no team is successful in solving all 1001 graphs by the competition deadline, then the team which solved the largest number of graphs will win the prize, so research teams are encouraged to submit partial sets of solutions early and often!

CHALLENGE STATUS: Finished!

Challenge Results

No teams were successful in solving all 1001 graphs, so the team with the best submission was awarded the prize. Congratulations to Nathann Cohen and David Coudert who were successful in finding Hamiltonian cycles in 985 graphs.

 

COMPETITION RULES

1) Individuals or teams may make multiple submissions, but only those solutions in their latest submission will be counted, so later submissions should also includes solutions discovered earlier.

2) Solutions will only be accepted until 11:59PM (GMT +9:30) on the 30th September, 2016.

3) The individual or team that solves the most graphs will be successful. In the event of a tie, the individual or team whose entry was received the earliest will be successful.

4) Solutions must be in the proper format (see below). If you are concerned, you should contact Michael Haythorpe with an example of a tour file to confirm that you are using the correct format.

5) Solutions should zipped in any standard archiving format (zip, rar, 7z, tar, gzip) and emailed to Michael Haythorpe.

6) Any queries or requests for clarification should be sent to Michael Haythorpe.

The FHCP Challenge Set competition was announced in the 59th Annual Meeting of the Australian Mathematical Society, held at Flinders University in September, 2015. The winners will be announced during the upcoming 60th Annual Meeting of the Australian Mathematics Society in 2016.

FORMATS

All graphs in the FHCP Challenge Set are in HCP format. A simple example of such a file is given below. The HCP file consists of a short header, an edge list, and a short footer.

Note that anything can be listed for NAME and COMMENT, while DIMENSION corresponds to the number of vertices. The other header and footer options should be unchanged.

NAME : Envelope graph
COMMENT : Example 6-vertex graph with HCs 1-2-3-6-5-4-1, 1-2-6-3-4-5-1 and 1-4-5-6-3-2-1
TYPE : HCP
DIMENSION : 6
EDGE_DATA_FORMAT : EDGE_LIST
EDGE_DATA_SECTION
1 2
1 4
1 5
2 3
2 6
3 4
3 6
4 5
5 6
-1
EOF

A tour file has a similar format, except instead of an edge list, it contains a list of vertices in the order visited in the Hamiltonian cycle, and the EDGE_DATA_FORMAT item is not required in the header.

NAME : Envelope tour file
COMMENT : Tour found
TYPE : TOUR
DIMENSION : 6
TOUR_SECTION
1
2
3
6
5
4
-1
EOF