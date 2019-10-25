# MonsterMirror
### An electronic funhouse mirror that changes people into animals and monsters

Set this up for your haunted house at Halloween!  Amuse your guests as their faces are dynamically transformed into different animals and monsters in front of their eyes.  Watch the animals' faces and expression change as yours does.  Feed it different example animals and monsters to see what it can generate.

## Status 

The source code for this project has not yet been released. This project is a placeholder while I work out a license I'm comfortable with.

## How does it work?

It uses modern computer vision techniques based in deep learning to first locate any faces in the image, and then feed them through an encoder-decoder network to modify them into a new structure. For details see [this video about the FUNIT model](https://www.youtube.com/watch?v=kgPAqsC8PLM&feature=youtu.be) or the [project page](https://nvlabs.github.io/FUNIT/).

## How can I use it?

Please respect people's privacy when using this system.  This software must only be used for live performances.  Recording or transmission of the created media is not allowed without express written consent.  Moreover, the display must be in clear view of the people being captured.  For details, see the [license](LICENSE).


## Acknowledgements

The key model in this software is the GANimal / FUNIT model from NVIDIA: [https://github.com/NVLabs/FUNIT](https://github.com/NVLabs/FUNIT) which is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Other important dependencies include:

* [SFD_Pytorch](https://github.com/clcarwin/SFD_pytorch)
* PyTorch
* Numpy
* OpenCV
* Python
