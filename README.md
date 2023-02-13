## Hippo

Hippo is a hierarchical latent diffusion guidance framework to reconstruct multi-grained activity sensing data according to different privacy requirements. With a deep hierarchical understanding of human motions, Hippo extracts different granularity of latent feature representations of activity sensing data. The latent features guide the reconstruction of multi-grained sensing data to preserve privacy.

The idea comes from lossy compression: (i) encoding, the file is converted to a more compact format; (ii) decoding, the original file is restored but some information is discarded. For example, the JBIG2 format extracts areas that look similar and stores all similar areas as one copy. This copy is reused to reconstruct the image. 

Similar to human activities, which have some common or similar behavioral characteristics. An activity can be decomposed into atomic actions, and the atomic actions are common or similar between different high-level activities. For example, the coarse-grained semantic category \textit{leg-based exercise} can be divided into fine-grained activities: walking, running and jumping. Taking a deeper look into jumping, it includes different subtle actions such as arms swinging, legs contracting, and stretching. The deep learning models and the diffusion models are shown to learn multi-resolution representations of the data as shown in Lasagna [Mobicom'16](https://dl.acm.org/doi/10.1145/2973750.2973752) and DDIM [ICLR'21](https://arxiv.org/pdf/2010.02502.pdf).

Here is a source code folder to show the implementation of Hippo. Especially, hippo4recon.py is used for training the model of Hippo on the harbox_sensys21_dataset, and hippo_gen_harbox.py is used for multi-grained data generation with a well-trained model. Note that the full datasets and full codes will be released upon the publication of the paper. We acknowledge that part of the code is built upon the open-source project diffuser.

