# ADI-Internship

During the internship in ADI, debayerization over CNN Accelerator is investigated.

As an initial step, literature search is done and useful papers are listed in the repo.

Several architectures are created in order to convert a bayer image to a 3-color image. The performance of the models are tested via pipelined models.

b2rgb.py is a relatively big model compared to other architectures. It takes a bayer image whose missing pixels are filled with zeros.

bayer2rgb.py consists of 3 different architectures. One can investigate several architectures by commenting out the necessary lines in the code. Even though the architectures differ from each other, all of them start with a bayerization followed by a folding operation.

All models are trained and the related parameters are given in the files.

With a pipeline architecture, several existing CNN models such that classification and segmentation with different datasets are tested. In the pipeline architecture, the first model converts bayer images to RGB images. As the second model, different architectures created by ADI AI Team are used.

The main purpose of the research is to find out whether bilinear interpolation is adequate for images coming from camera or another approach can be adopted that improves not only accuracy but also power & speed.

More detailed explanation of the studies can be found in the presentation. 
