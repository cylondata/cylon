---
id: 0.6.0
title: Cylon Release 0.6.0
sidebar_label: Cylon Release 0.6.0
---

Cylon 0.6.0 is a major release. We are excited to present UCC, Gloo integration, More distributed operations

## Features 

### Cylon C++ and Python
- Implemention of Slice, Head and Tail Operations
- adding conda docker
- Ucc integration
- adding cylonflow as a submodule
- Use generic operator 
- Summit fixes 
- Adding custom mpirun params cmake var
- Adding cmake parallelism flag
- Gloo python binding
- Enabling gloo CI
- Add downloading catch2 header dynamically
- Dist sort cpu 
- Cylon Gloo integration
- Adding distributed scalar aggregates
- Extending datatypes
- Allowing custom MPI_Comm for MPI


### Build
- Updating to Arrow 0.9.x
- Windows build support
- MacOS build support
- Conda build is the default build
- Improving docker build


You can download source code from [Github](https://github.com/cylondata/cylon/releases)
Conda binaries are available in [Anaconda](https://anaconda.org/cylondata)

## Commits
91bdd54 Update conda-actions.yml (#645)
d1739ed Added buildable instructions for Rivanna (#643)
d9a6420 Arrow 9.0.0 and gcc-11 update  (#601)
4c867b1 Summit Fixes  (#623)
7f8a3b1 Fixing sample bug (#631)
ce12454 Cython binding for slice, head and tail (#619)
ef4c904 #610: SampleArray util method replaced by using arrow::compute::Take … (#612)
4694a9e Minor fixes (#608)
121b386 Fixing: Corrupted result when joining tables contain list data types #615  (#616)
68fa598 Summit fixes  (#607)
de3ec7b fixing bash splitting (#606)
0a489fc adding cmake parallelism flag (#605)
035fd70 Implement Slice, Head and Tail Operation in both centralize and distr… (#592)
d99a6f2 adding custom mpirun params cmake var (#604)
f20c119 Update README-summit.md (#603)
4bc27f9 Create README-summit.md (#602)
e6b7306 Minor fixes  (#596)
2e6ac80 adding conda docker (#600)
4dd359f Ucc integration (#591)
61b4a82 adding cylonflow as a submodule (#593)
e4dd38b Use generic operator (#583)
6c0dfa8 Gloo python binding (#587)
773f11f Gloo python bindings (#585)
2fc95be Add downloading catch2 header dynamically (#584)
c56ab2d Enabling gloo CI (#582)
a820ed8 Dist sort cpu (#574)
f68cc62 Adding UCC build (#579)
2759a30 Cylon Gloo integration (#576)
b2c0820 Adding distributed scalar aggregates (#570)
9c2fdc4 Extending datatypes (#568)
e3d553c Bump ua-parser-js from 0.7.22 to 0.7.31 in /docs (#566)
3bafb75 Bump ssri from 6.0.1 to 6.0.2 in /docs (#565)
814a463 minor fixes (#564)
be92253 Bump lodash from 4.17.20 to 4.17.21 in /docs (#561)
e87dd7c Bump shelljs from 0.8.4 to 0.8.5 in /docs (#562)
71bd8bf Bump nanoid from 3.1.22 to 3.2.0 in /docs (#563)
49b343d Allowing custom MPI_Comm for MPI (#559)
fa52dd4 Update contributors.md
54d4a53 added io functions (#550)
1a8c3d7 Fixing 554  (#558)
887ea18 update arrow link (#557)
1ce4c6b Fixing 552 (#553)
f5e31a1 Merging 0.5.0 release  (#547)

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
Arup Kumar Sarker
Mills Wellons Staylor


## License

Licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0