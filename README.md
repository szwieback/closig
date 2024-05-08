# closig

Temporal closure signatures from interferometric covariance matrices. A closure signature is determined using a basis of the annihilator of consistent phases, i.e., a list of linear functionals that measure the deviation from consistency. Bases with meaningful time-timescale characteristics enable extraction of temporal closure signatures that lend themselves to interpretation and comparison to models. 

The key functionalities are:
* extraction of temporal closure signatures using two complementary bases (small steps and two hops)
* visualization of closure signatures
* aggregation of closure signatures in time and space
* phase linking comparison experiments; phase linking functionality is provided through the greg package
* modeled closure signatures from simple interferometric scattering models

The code is modular to support scientific inquiries but is inefficient. It is not intended for large-scale processing.

## Requirements

Tested on Python 3.10.4, with numpy 1.22.3, matplotlib, abc, pickle, zlib. The greg package is needed for phase linking.

## Contact

[Simon Zwieback](https://szwieback.github.io)
[Rowan Biessel](https://github.com/rbiessel)

Feedback, ideas and contributions are greatly appreciated.

## License

This project is licensed under the MIT license.

