# Introduction to NeRF and Its Key Developments


NeRF stands for Neural Radiance Fields. It solves for view interpolation, which is taking a set of input views (in this case a sparse set) and synthesizing novel views of the same scene. Current RGB volume rendering models are great for optimization, but require extensive storage space (1-10GB). One side benefit of NeRF is the weights generated from the neural network are ~6000 less in size than the original images.

# Helpful Terminology

- **Rasterization**: Computer graphics use this technique to display a 3D object on a 2D screen. Objects on the screen are created from virtual triangles/polygons to create 3D models of the objects. Computers convert these triangles into pixels, which are assigned a color. Overall, this is a computationally intensive process.
- **Ray Tracing**: In the real world, the 3D objects we see are illuminated by light. Light may be blocked, reflected, or refracted. Ray tracing captures those effects. It is also computationally intensive, but creates more realistic effects.
- **Ray**: A ray is a line connected from the camera center, determined by camera position parameters, in a particular direction determined by the camera angle.\
- **NeRF uses ray tracing rather than rasterization for its models.**
- **Volume Rendering**: This is a technique used to display a 2D projection of a 3D discretely sampled data set. A typical example is a CT scan. The data is a 3D array of voxels, and the rendering is done by projecting the voxels onto a 2D plane.
- **Neural Rendering** As of 2020/2021, this terminology is used when a neural network is a black box that models the geometry of the world and a graphics engine renders it. Other terms commonly used are *scene representations*, and less frequently, *implicit representations*. In this case, the neural network is just a flexible function approximator and the rendering machine does not learn at all.

# Approach
![ne](https://github.com/karanamrahul/Sidehustler/blob/main/D_03_NeRF/3comp.png)


A continuous scene is represented as a 3D location *x* = (x, y, z) and 2D viewing direction $(\theta,\phi)$ whose output is an emitted color c = (r, g, b) and volume density $\sigma$. The density at each point acts like a differential opacity controlling how much radiance is accumulated in a ray passing through point *x*. In other words, an opaque surface will have a density of $\infty$ while a transparent surface would have $\sigma = 0$. In layman terms, the neural network is a black box that will repeatedly ask what is the color and what is the density at this point, and it will provide responses such as “red, dense.”\
This neural network is wrapped into volumetric ray tracing where you start with the back of the ray (furthest from you) and walk closer to you, querying the color and density. The equation for expected color $C(r)$ of a camera ray $r(t) = o + td$ with near and far bounds $t_n$ and $t_f$ is calculated using the following:

$$
C(r) = \int_{t_n}^{t_f} T(t)\sigma(r(t))c(r(t),d) \,dt
$$
where
$$
T(t) = exp\left(-\int_{t_n}^{t}\sigma(r(s))\, ds\right)
$$


To actually calculate this, the authors used a stratified sampling approach where they partition $[t_n, t_f]$ into N evenly spaced bins and then drew one sample uniformly from each bin:

$$\hat{C}(r) = \sum_{i = 1}^{N}T_{i}(1-exp(-\sigma_{i}\delta_{i}))c_{i}, where T_{i} = exp(-\sum_{j=1}^{i-1}\sigma_{j}\delta_{j})$$

![eq](https://github.com/karanamrahul/Sidehustler/blob/main/D_03_NeRF/eq.png)
Where $\delta_{i} = t_{i+1} - t_{i}$ is the distance between adjacent samples. The volume rendering is differentiable. You can then train the model by minimizing rendering loss.

$$min_{\theta}\sum_{i}\left\| render_{i}(F_{\Theta}-I_{i}\right\|^{2}$$

![NeRF](https://github.com/karanamrahul/Sidehustler/blob/main/D_03_NeRF/paper.png)
In this illustration taken from the paper, the five variables are fed into the MLP to produce color and volume density. $F_\Theta$ has 9 layers, 256 channels.

In practice, the Cartesian coordinates are expressed as vector d. You can approximate this representation through MLP with $F_\Theta = (x, d) \rightarrow (c, \sigma)$.\
**Why does NeRF use MLP rather than CNN?**
Multilayer perceptron (MLP) is a feed forward neural network. The model doesn’t need to conserve every feature, therefore a CNN is not necessary.\

# Common issues and mitigation

The naive implementation of a neural radiance field creates blurry results. To fix this, the 5D coordinates are transformed into positional encoding (terminology borrowed from transformer literature). $F_\Theta$ is a composition of two formulas: $F_\Theta = F'_\Theta \cdot \gamma$ which significantly improves performance.

$$\gamma(p) = (sin(2^{0}\pi p), cos(2^{0}\pi p),...,sin(2^{L-1}\pi p), cos(2^{L-1} \pi p)$$

L determines how many levels there are in the positional encoding and it is used for regularizing NeRF (low L = smooth). This is also known as a Fourier feature, and it turns your MLP into an interpolation tool. Another way of looking at this is your Fourier feature based neural network is just a tiny look up table with extremely high resolution. Here is an example of applying Fourier feature to your code:

```
B = SCALE * np.random.normal(shape = (input_dims, NUM_FEATURES))
x = np.concatenate([np.sin(x @ B), np.cos(x @ B)], axis = -1)
x = nn.Dense(x, features = 256)
```
![Fourier Features](https://github.com/karanamrahul/Sidehustler/blob/main/D_03_NeRF/fourier.png)

Mapping how Fourier features are related to NeRF’s positional encoding. Taken from Jon Barron’s CS 231n talk in Spring 2021.

NeRF also uses hierarchical volume sampling: coarse sampling and the fine network. This allows NeRF to more efficiently run their model and deprioritize areas of the camera ray where there is free space and occlusion. The coarse network uses $N_{c}$ sample points to evaluate the expected color of the ray with the stratified sampling. Based on these results, they bias the samples towards more relevant parts of the volume.

$$\hat{C}_c(r) = \sum_{i=1}^{N_{c}}w_{i}c_{i}, w_{i}=T_{i}(1-exp(-\sigma_{i}\delta_{i}))$$

A second set of $N_{f}$ locations are sampled from this distribution using inverse transform sampling. This method allocates more samples to regions where we expect visual content.

### Key Developments in NeRF

Neural Radiance Fields (NeRF) have significantly advanced the field of neural rendering, enabling the creation of photorealistic 3D models from a set of 2D images. Here's a summary of pivotal papers from the inception of NeRF in 2020 through various advancements up to 2023, highlighting their key contributions and innovations:

## 2020: The Beginning of NeRF

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**
  - **Key Contributions:** Introduced the concept of NeRF, using a Multilayer Perceptron (MLP) to model volumetric scene functions from sparse input images, enabling high-quality 3D renderings.NeRF models a scene using a function \(F(\mathbf{x}, \mathbf{d}) = (\mathbf{c}, \sigma)\), where \(\mathbf{x}\) is a point in space, \(\mathbf{d}\) is the viewing direction, \(\mathbf{c}\) is the emitted color, and \(\sigma\) is the density at \(\mathbf{x}\). This function is approximated using a Multilayer Perceptron (MLP).
  - **Mathematical Insight:** NeRF models the scene as a function $f(x, y, z, \theta, \phi)$ mapping a 3D position and 2D viewing direction to color and density.
  By integrating the radiance along camera rays using volume rendering techniques, NeRF can synthesize novel views with intricate details and lighting effects.
  - [Paper](https://arxiv.org/abs/2003.08934) | [Demo Site](https://www.matthewtancik.com/nerf) | [GitHub](https://github.com/bmild/nerf) | [Explainer Video](https://www.youtube.com/watch?v=CRlN-cYFxTk&t=2s)

- **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**
  - **Key Contributions:** Demonstrated that applying Fourier feature transformations to inputs allows MLPs to capture high-frequency details more effectively.
  - **Mathematical Insight:** Utilizes the transformation $\gamma(x) = [\cos(2\pi Bx), \sin(2\pi Bx)]$, where $B$ is a matrix of frequencies.Fourier features enable the network to better model fine textures and edges, crucial for photorealism.This paper enhances NeRF's ability to capture high-frequency details by mapping inputs through a Fourier feature transformation. This process allows the MLP to learn high-frequency functions more effectively.
  - [Paper](https://arxiv.org/abs/2006.10739) | [Demo Site](https://www.youtube.com/watch?v=nVA6K6Sn2S4) | [GitHub](https://github.com/tancik/fourier-feature-networks)

## 2021: Expanding the Boundaries

- **NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections**
  - **Key Contributions:** Extended NeRF to handle unconstrained photo collections, introducing latent variables for appearance and transient objects.
  - Extended NeRF to handle unconstrained input photo collections, improving versatility in real-world applications.
  - Introduced techniques to decouple appearance data from geometry, enhancing adaptability to diverse lighting and environmental conditions.
  - **Transient Objects:** NeRF-W distinguishes between static and transient aspects of a scene. It uses a dual-head architecture where one head models the permanent structure of the scene, and another models transient phenomena, such as people or cars. The system can thus ignore or down-weight the influence of these transient objects during the view synthesis process.
  - **Mathematical Insight:** Incorporates image-dependent appearance embeddings to model photometric variations and transient phenomena separately.
  - [Paper](https://arxiv.org/abs/2008.02268) | [Demo Site](https://nerf-w.github.io/) | [Explainer Video](https://www.youtube.com/watch?v=mRAKVQj5LRA)

- **Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields**
  - **Key Contributions:** Improved handling of aliasing and resolution changes by incorporating a multiscale representation through feature mapping.
  - Tackled the issues of scale and aliasing in NeRF, which struggled with changing resolutions or scales.
  - Implemented a mipmap-inspired feature mapping, achieving superior anti-aliasing results.
  - **Mathematical Insight:** Adapts the feature mapping to create an image pyramid, enhancing the model's capability to deal with varying scales and resolutions.
  - [Paper](https://arxiv.org/abs/2103.13415) | [Demo Site](https://jonbarron.info/mipnerf/) | [GitHub](https://github.com/google/mipnerf)

- **D-NeRF: Neural Radiance Fields for Dynamic Scenes**
  - **Key Contributions:** Adapted NeRF for dynamic scenes by modeling temporal changes, allowing for the rendering of animations and deformable objects.
  - **Dynamic Scene Modeling:** D-NeRF introduces a methodology for modeling dynamic scenes by extending the static NeRF model. It incorporates time as an additional input dimension to the neural network, enabling the model to capture temporal variations in the scene.

  - **Deformation Field:** To manage the dynamic aspects of the scene, D-NeRF proposes a deformation field that maps each point in the scene at any given time to a canonical, undeformed state. This approach allows the model to learn the dynamic behavior of the scene effectively.

  - **Time-conditioned Rendering:** By conditioning the radiance field on time, D-NeRF can render images of the scene at any given moment, capturing the movement and changes that occur over time.

  - **Mathematical Insight:** Uses two neural networks to map scene dynamics over time, transforming points back to a canonical reference frame.
  - [Paper](https://arxiv.org/abs/2011.13961) | [Demo Site](https://www.albertpumarola.com/research/D-NeRF/index.html) | [GitHub](https://github.com/albertpumarola/D-NeRF)

- **PixelNeRF: Neural Radiance Fields from One or Few Images**
  - **Key Contributions:** Demonstrated the feasibility of generating NeRF models from minimal input, using as few as one image, by leveraging a pre-trained CNN as a prior.
  - **Efficient Scene Reconstruction from Sparse Views:** PixelNeRF can reconstruct scenes from a significantly smaller set of images compared to the original NeRF, even managing to perform well with just a single image in some cases.

  - **Integration of 2D Convolutional Neural Networks (CNNs):** The method integrates learned features from 2D CNNs with the volumetric scene representation used in NeRF. This allows pixelNeRF to leverage the powerful image understanding capabilities of CNNs to inform the 3D reconstruction process.

  - **Generalization to Unseen Scenes:** Unlike the original NeRF, which is scene-specific and requires training a separate model for each new scene, pixelNeRF demonstrates an ability to generalize across scenes. This means it can be applied to new scenes without needing retraining from scratch, significantly reducing the computational cost and time required to model new environments.

  - **Conditional Rendering:** PixelNeRF enables conditional rendering based on a small number of input images. It can predict the appearance of a scene from novel viewpoints not covered by the input images, showcasing a robust understanding of the scene's 3D structure.
  - **Mathematical Insight:** Combines a conventional NeRF network with a pretrained CNN, adjusting the weighting based on the similarity between the input and desired views.
  - [Paper](http://arxiv.org/abs/2012.02190) | [Demo Site](https://alexyu.net/pixelnerf/) | [GitHub](https://alexyu.net/pixelnerf/)

- **RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs**
  - **Key Contributions:** Targets the challenge of sparse inputs by introducing regularization techniques and generating additional viewpoints for training.

  - **Regularization Techniques:** RegNeRF introduces novel regularization techniques that are specifically designed to improve the quality of NeRF reconstructions from sparse inputs. These techniques help in mitigating overfitting to the sparse input images and enhance the model's ability to infer unseen parts of the scene.

  - **Improved Handling of Sparse Views:** By focusing on scenarios with limited input data, RegNeRF significantly improves the fidelity of view synthesis compared to the original NeRF model. This is particularly valuable in contexts where only a few images of a scene are available.

  - **Enhanced Generalization:** The regularization strategies employed by RegNeRF not only improve performance on the given sparse inputs but also enhance the model's generalization capabilities to unseen views. This is crucial for practical applications of view synthesis, where generating new, plausible views of a scene is often the primary goal.

  - **Quality of Synthesized Views:** RegNeRF demonstrates an ability to maintain high quality in synthesized views even when the input data is sparse. This includes better handling of fine details and textures, which are often challenging to capture with limited information.

  - **Mathematical Insight:** Employs regularization on geometry and color from synthesized viewpoints to enhance stability and fidelity with limited data.
  - [Paper](https://arxiv.org/abs/2112.00724) | [Demo Site](https://m-niemeyer.github.io/regnerf/) | [GitHub](https://github.com/google-research/google-research/tree/master/regnerf)


### KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs

- Addressed the computational inefficiency of NeRF, replacing the large MLP with thousands of tiny MLPs for faster rendering.
- Leveraged a voxel grid subdivision and teacher-student distillation to maintain visual quality while enhancing speed.

## 2022: Pushing Boundaries Further

### Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

- Optimized NeRF for unbounded scenes and 360° imagery, incorporating non-linear scene parameterization and novel regularizers.
- Introduced online distillation and distortion-based regularization to handle infinite depths and improve anti-aliasing.

### TensoRF: Tensorial Radiance Fields

- Represented radiance fields as a 4D tensor, decomposing into matrix and vector components for improved efficiency and quality.
- Achieved faster reconstruction times and superior visual results with a minimal memory footprint.

### Block-NeRF: Scalable Large Scene Neural View Synthesis

- Generated city-scale radiance fields from millions of images, training individual NNs for coherent synthesis across vast areas.
- Combined techniques from NeRF in the Wild and Mip-NeRF for seamless integration and interpolation of city blocks.

## 2023: The Latest Advancements

### Refinements and New Directions

- The latest advancements, such as **NeRF in the Dark, Zip-NeRF, RUST, MobileNeRF,** and **K-Planes** continued to refine and expand the capabilities of NeRF, tackling challenges like high dynamic range imaging, efficient rendering on mobile architectures, and explicit radiance field representations.
- **Bayes' Rays** introduced uncertainty quantification, while **Plenoxels** offered a non-neural network approach to radiance fields, emphasizing efficiency and simplicity.
- **3D Gaussian Splatting** focused on real-time rendering improvements, showcasing the ongoing innovation in the field.

## Conclusion: A Future Unbounded

The journey through NeRF's mathematical innovations and technical breakthroughs showcases the field's rapid evolution. Each development, from foundational concepts to the latest refinements, extends the boundaries of what's possible in 3D content creation. As we venture further, the fusion of deep learning with computer graphics continues to unlock new realms of realism, efficiency, and creativity in virtual and augmented reality, filmmaking, and beyond.


## Credits and References

This overview of Neural Radiance Fields (NeRF) and its developments is based on a series of groundbreaking research papers. Below are the references to the original works that have significantly contributed to the advancements in the field of 3D scene reconstruction and rendering using NeRF technology.

### Foundational Papers

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**
  - Mildenhall, Ben, et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020*.

- **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**
  - Tancik, Matthew, et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *NeurIPS 2020*.

### Key Advancements in 2021

- **NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections**
  - Martin-Brualla, Ricardo, et al. "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections." *CVPR 2021*.

- **Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields**
  - Barron, Jonathan T., et al. "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." *ICCV 2021*.

- **D-NeRF: Neural Radiance Fields for Dynamic Scenes**
  - Pumarola, Albert, et al. "D-NeRF: Neural Radiance Fields for Dynamic Scenes." *CVPR 2021*.

- **PixelNeRF: Neural Radiance Fields from One or Few Images**
  - Yu, Alex, et al. "pixelNeRF: Neural Radiance Fields from One or Few Images." *CVPR 2021*.

- **RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs**
  - Niemeyer, Michael, et al. "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs." *CVPR 2021*.

- **KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs**
  - Reiser, Christian, et al. "KiloNeRF: Speeding up Neural Radiance Fields with Thousands of Tiny MLPs." *ICCV 2021*.

### Pushing Boundaries Further in 2022

- **Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields**
  - Barron, Jonathan T., et al. "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." *SIGGRAPH 2022*.

- **TensoRF: Tensorial Radiance Fields**
  - Chen, Vincent Sitzmann, et al. "TensoRF: Tensorial Radiance Fields." *SIGGRAPH 2022*.

- **Block-NeRF: Scalable Large Scene Neural View Synthesis**
  - Tancik, Matthew, et al. "Block-NeRF: Scalable Large Scene Neural View Synthesis." *SIGGRAPH 2022*.

### Latest Advancements in 2023

- Further details and references for 2023 advancements such as *NeRF in the Dark*, *Zip-NeRF*, *RUST*, *MobileNeRF*, *K-Planes*, *Bayes' Rays*, *Plenoxels*, and *3D Gaussian Splatting* can be found in their respective publications and project pages.

## Blogs

- [Richard Skarbez's Blog](https://www.richardskarbez.com/nerf-tutorial) - A comprehensive tutorial on NeRF and its applications.
- [CS231n Course](https://github.com/cs231n/cs231n.github.io/tree/master) - CS231n Lecture Notes.

