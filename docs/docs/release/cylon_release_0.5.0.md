---
id: 0.5.0
title: Cylon Release 0.5.0
sidebar_label: Cylon Release 0.5.0
---

Cylon 0.5.0 is a major release. We are excited to present **GCylon**, cudf-based distributed 
DataFrame for Nvidia GPUs, UCX integration, Anaconda support, and much more. 

## Features 

### Cylon C++ and Python
- Adding UCX integration with MPI
- Adding read distribution
- Changing join column naming convention to match SQL and pandas
- Adding `Dataframe.applymap`, `Dataframe.isin`
- Add iloc operation to DataFrame
- Adding null handling to table operators and Comparators
- Adding Equal/ distributed equal operators
- Adding array flattening
- Adding Repartition
- Adding mapreduce style group-by aggregators
- Adding table level AllGather, Gather and Broadcast operators
- Performance improvements and bug fixes


### Build
- Updating to Arrow 0.5.x
- Windows build support
- MacOS build support
- Conda build is the default build
- Improving docker build

### Gcylon
First release of Gcylon which supports distributed DataFrame processing on Nvidia GPUs using CuDF:
- Implemented shuffling and distributed sorting
- Distributed Join/merge
- Distributed GroupBy
- DataFrame Set operations
- Repartitioning DataFrames
- Distributed IO for reading/writing CSV, JSON and Parquet files


You can download source code from [Github](https://github.com/cylondata/cylon/releases)
Conda binaries are available in [Anaconda](https://anaconda.org/cylondata)

## Commits

3344bf95 Mapreduce style group-by aggregators  (#535)
50ef890b Remove minor warnings (#544)
559e8eb3 Adding CPU serializer (#539)
abb44049 fixed unused variable/parameter and casting warnings (#542)
62a3f080 Distributed IO (#533)
15d06d6c Bump color-string from 1.5.4 to 1.7.4 in /docs (#534)
810c4ed7 fixing RNG issue (#538)
fbb049bb fixing build error (#536)
a10e0528 Bump algoliasearch-helper from 3.3.3 to 3.6.2 in /docs (#532)
112ea97f Repartition - CPU (#526)
79c4b739 create a MacOS yml file (#530)
b9e7a8c4 Repartition - GPU (#528)
2191b9f5 fixed function name change in cudf api from gcylon test files (#529)
3e9036ee Upgrading to arrow 5.0.0 (#525)
24d182ab Groupby values null handling  (#527)
54a5074b Null handling for Comparators  (#524)
0b9516e7 Adding array flattening  (#522)
b3fc2a2a Implemented MergeOrSort when merging sorted tables (#523)
1e061b2f Feature/equal (#499)
e378d1dc reformatted gcylon codes with tab size 2, non-functional changes (#521)
8450d9b1 Added support for sliced tables in gather, broadcast and sorting (#520)
92b8124c Update windows.yml
1f9790d7 Update macos.yml
d33f9ac8 Update conda-actions.yml
963d4914 Update c-cpp.yml
2229981d added mpi datatype dispatching for primitive data types (#519)
d9936b4d Head tail operators (#512)
ac99d009 Formatting code (#518)
fff84ccb Code formatting  (#517)
f32f04da Null handling in splitters and build arrays (#511)
4cab7ca4 Delete files from CPP example folder that are not needed (#516)
d1744302 moving tutorial repo to (#514)
9cd7911f Python example cleanup (#513)
fe4caf37 Distributed sorting (#510)
2302f58f Minor improvements to the Table API (#508)
71eb80a1 adding new test utils (#507)
24b83dd3 Adding to docker docs (#498)
6f2faf8f Update conda.md
4f8f3c7f Gcylon docs (#501)
a7862580 Adding contributing guide to documentation (#496)
8ab8b2d6 changing join column naming convention to match SQL and pandas (#487)
f18b91fe improvements to ucx build from conda (#484)
912fb543 Windows build (#482)
216758a2 making improvements to the build (#483)
4e2894eb Add functions to dataframe (#481)
1f1ddd9c Documentation update (#479)
e6233151 Bump tar from 6.1.5 to 6.1.11 in /docs (#477)
1e5db7b6 improve docs (#476)
58c0595d removing extra examples (#474)
3c823f6f Gcylon integration (#470)
92748eb5 Cpp example cleanup (#475)
fa14527d Docs improvements (#469)
13062206 Bump url-parse from 1.4.7 to 1.5.3 in /docs (#473)
8234ae7b Bump path-parse from 1.0.6 to 1.0.7 in /docs (#472)
c8b435b6 Bump tar from 6.0.5 to 6.1.5 in /docs (#471)
1cc28dd3 Performance improvements (#453)
9092bbf0 MacOS build (#464)
d59d91ea Add iloc operation to DataFrame (#465)
8d7a8dc7 Removed glog files from the header files (#463)
ea62eef0 License updates (#462)
2f562650 changed all relative Cylon header references to global (#461)
123c93c3 Building in conda env without using conda-build (#457)
3b3a2853 Compilation document improvements (#454)
8578b1f1 Adding barrier at the end of the test case (#458)
e6eded5f Fix for empty df (#455)
8f149924 Fixed mpi test case (#456)
cb069980 Changes to the Docs (#451)
4ce1d7eb updates to the docker readme
e011e0f6 enhancing readme
adfa6c05 adding read distribution (#432)
bd2e024d UCX integration (#439)
a42d04ad Bump ws from 6.2.1 to 6.2.2 in /docs (#437)
710b562e Bump dns-packet from 1.3.1 to 1.3.4 in /docs (#435)
07aee740 adding new operators to DataFrame API (#429)
71e57f84 Updating to arrow 4.0 (#418)
a490dc21 changing ctx to const reference in methods (#419)
18a5447b missing docs (#428)
38534f55 0.4.1 release (#427)
10f5a6a3 Enabling scalars in df set_item (#425)
0be78972 Op bench refactor (#417)
ec964d89 Bug fixes in dataframe  (#420)
e0ba9643 Update c-cpp.yml
0200c021 adding finalize check and removing destructor finalize call. (#412)
149919c2 Update README.md
016c5c92 adding missing test case
56095357 Update README.md
e3ca0bf5 0.4.0 release (#411)

## Contributors

Ahmet Uyar
Chathura Widanage
Damitha Sandeepa Lenadora
dependabot[bot]
Hasara Maithree
Kaiying Shan
niranda perera
Supun Kamburugamuve
Vibhatha Lakmal Abeykoon
Ziyao22


## License

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0