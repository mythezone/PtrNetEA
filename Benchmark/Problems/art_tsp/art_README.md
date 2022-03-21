	


Robert Bosch has created a fascinating series of instances of the traveling salesman problem (TSP) that provide continuous-line drawings of well-known pieces of art. Techniques for developing such point sets have evolved over the past several years through work of Bosch and Craig Kaplan.

One of Bosch's instances is the 100,000-point set for the Mona Lisa TSP Challenge. Additional instances range in size up to 200,000 cities, providing a difficult test for TSP solution methods. The data sets are specified in TSPLIB format. We thank Bob for making these beautiful problems available to the research community.

Original Art	Data Set		Cities	

da Vinci's Mona Lisa	mona-lisa100K.tsp		100,000

van Gogh's Self Portrait 1889	vangogh120K.tsp		120,000

Botticelli's The Birth of Venus	venus140K.tsp		140,000

Velazquez's Juan de Pareja	pareja160K.tsp		160,000

Courbet's The Desperate Man	courbet180K.tsp		180,000

Vermeer's Girl with a Pearl Earring	earring200K.tsp		200,000


	
mona-lisa100K.tsp
vangogh120.tsp
	
venus140K.tsp
pareja160K.tsp
	
courbet180K.tsp
earring200K.tsp
The following papers discuss the mathematics behind the selection of city locations for these TSP Art instances.

Robert Bosch, Opt Art, Math Horizons, February 2006, pages 6--9.
Craig Kaplan and Robert Bosch, TSP Art, Renaissance Banff: Bridges 2005: Mathematical Connections in Art, Music and Science, pages 301-308.
Robert Bosch and Adrianne Herman, Continuous Line Drawings via the Traveling Salesman Problem, Operations Research Letters, 2004, Volume 32, pages 302--303.
Ivars Peterson, Artful Routes, Mathematical Association of America Online, January 3, 2005.
Pretty examples of other TSP drawings can be found on Robert Bosch's TSP Art page and on the page of Craig Kaplan.

The best known results for the TSP Art instances are given in the table below. The tour length, given in the Best Tour column, is a link to the tour in TSPLIB format. I would be happy to post any improvements you find.

Problem	Best Tour		Source of Tour	

mona-lisa100K	5,757,191		Yuichi Nagata (2009)

vangogh120K	6,543,609		Kazuma Honda, Yuichi Nagata, Isao Ono (2013)

venus140K	6,810,665		Yuichi Nagata (2011)

pareja160K	7,619,953		Yuichi Nagata (2011)

courbet180K	7,888,731		Kazuma Honda, Yuichi Nagata, Isao Ono (2013)

earring200K	8,171,677		Yuichi Nagata (2011)


Notes

The vangogh120k and courbet180k tours were received from Yuichi Nagata on July 16, 2019. The tours are reported in the paper A Parallel Genetic Algorithm with Edge Assembly Crossover for 100,000-City Scale TSPs from 2013 by Honda, Nagata, and Ono. The previous best tours for these instances, found by Yuichi Nagata in 2011, had lengths 6,543,610 and 7,888,733, respectively. 

Thanks to Emilio Bendotti, Francesco Cavaliere and Matteo Fischetti for pointing me to the Honda et al. paper. The Bendotti-Cavaliere-Fischetti team themselves found a tour of length 7,888,732 for courbet180k on July 13, 2019, starting from Nagata's 7,888,733 tour.