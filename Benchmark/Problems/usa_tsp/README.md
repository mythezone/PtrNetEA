The Mona Lisa TSP Challenge was set up in February 2009. An optimal solution to that 100,000-city instance would set a new world record for the traveling salesman problem. It is a beautiful point set, but I have received many comments suggesting it would be nice to also keep in the spirit of actual salesmen and consider a similarly large challenge instance involving a trip through actual towns and cities.

Making use of the great data collected by the US Geological Survey, I'd like to propose a 115,475-city challenge through (nearly) all cities, towns, and villages in the contiguous 48 states. The nearly comes from pruning the USGS data set to remove places with different names but almost identical latitude-longitude coordinates.

A drawing of the point set is given below. Click on the image to see a larger version of the drawing. It is interesting to see how the locations of the towns follow natural and man-made structures in the midwest and west of the country.


USA Data Set
115,475 Towns and Cities in the United States, July, 2012
usa115475.tsp, usa115475.pdf
It is tempting to consider actual driving distances for the TSP instance, but this would make it a moving target (as roads are created or improved) and it would also make it very difficult to manage the large amount of data. To make the instance precise, we specify the travel cost between two points to be the Euclidean distance rounded to the nearest integer value, that is, we imagine our salesman owns a helicopter. This is the TSPLIB travel cost that has been the standard since the early 1990s.

The point set is available in TSPLIB format in the file usa115475.tsp.

Are there really 115,475 cities in the United States?
Previous TSP tours of the 48 states.
Hitting each of the 48 states.
To add a bit of spice, I offered $500 to the person/team who first provided, before July 4, 2013, the best tour reported up to that date. This is not asking for a provably optimal solution to the problem (although that is, of course, the long-term goal), but a year-long competition for tour-finding heuristic methods. Thanks to all of the researchers who responded to the challenge. There was a three-way tie between Xavier Clarist, Keld Helsgaun, and Yuichi Nagata, who each submitted a tour of length 6,204,999. Xavier wins the tiebreaker, since I received his tour on August 27, 2012, Keld's tour on September 17, 2013, and Yuichi's tour on February 1, 2013. Check the Leader Board for a full list of the tours received.