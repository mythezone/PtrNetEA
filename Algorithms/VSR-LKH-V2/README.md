# VSR-LKH(-3)
----
This repository contains the code to the VSR-LKH algorithm for the TSP proposed in our paper: <br> <br>
[Reinforced Lin-Kernighan-Helsgaun Algorithms for the Traveling Salesman Problems](https://ojs.aaai.org/index.php/AAAI/article/view/17476) <br>
Jiongzhi Zheng, Kun He, Jianrong Zhou, Yan Jin, Chu-Min Li <br> <br>

VSR-LKH
----
On a Unix/Linux machine execute the following commands: <br> <br>

unzip VSR-LKH-V2-main.zip <br>
cd VSR-LKH-V2-main <br>
chmod +777 -R VSR-LKH-Final <br>
cd VSR-LKH-Final <br>
make <br> <br>

An executable file called LKH will now be available in the directory VSR-LKH-Final. Then execute ./run_NAME_ALPHA (./run_NAME_POPMUSIC) to calculate the instance NAME.tsp by VSR-LKH with the α-measure (POPMUSIC) method. <br> <br>

VSR-LKH-3
----
On a Unix/Linux machine execute the following commands: <br> <br>

unzip VSR-LKH-V2-main.zip <br>
cd VSR-LKH-V2-main <br>
chmod +777 -R VSR-LKH-3-Final <br>
cd VSR-LKH-3-Final <br>
make <br> <br>

An executable file called LKH will now be available in the directory VSR-LKH-3-Final. Then place the instances in [VSR-LKH-3-Final/Instances](./VSR-LKH-3-Final/Instances) to the directory VSR-LKH-3-Final, and execute the following commands: <br> <br>

./run_TSPTW III NAME 10000 10 0 <br> <br>

or <br> <br>

./run_CTSP III NAME 10000 10 0 <br> <br>

to calculate the instance NAME.tsptw or NAME.ctsp by VSR-LKH-3 with the default settings in the paper. <br> <br>

Execute commands ./runAll_TSPTW or ./runAll_CTSP to calculate all the TSPTW or CTSP instances. <br> <br>

The default version of VSR-LKH-3 uses the α-measure to decide the candidate sets. Set the parameter CandidateSetType = POPMUSIC in [ReadParameters.c](./VSR-LKH-3-Final/SRC/ReadParameters.c) to get the VSR-LKH-3 with the POPMUSIC method. <br> <br>

Data
----
All the 236 tested TSP instances from TSPLIB, National TSP, VLSI TSP benchmarks are available in [Instances](./Instances). The parameter files for 236 TSP instances are available in [ParFiles](./ParFiles). All the 425 tested TSPTW instances and 65 tested CTSP instances are available in [VSR-LKH-3-Final/Instances](./VSR-LKH-3-Final/Instances). <br> <br>

Contact
----
Questions and suggestions can be sent to jzzheng@hust.edu.cn.
