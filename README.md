# sl2z-gan

This is a little toy project that experiments with using a generative adversarial network (GAN) to generate random matrices in $SL(2,\mathbb{Z})$.

# Learning goals

* Learn how to implement a GAN in PyTorch.

* Write clean and performant PyTorch & Lightning code.

* Discover some neat math along the way.

# Theory

## Random unimodular matrices

Let $\mathbb{Z}$ denote the ring of integers. For $n\geq 1$, we denote by $SL(n,\mathbb{Z})$ the set of $n\times n$ matrices over $\mathbb{Z}$ with determinant $1$. $SL(n,\mathbb{Z})$ forms a group under matrix multiplication, because an integer matrix has integer inverse if and only if its determinant is $\pm 1$, a consequence of the relationship between the adjugate matrix and the determinant. It is a subgroup of the *general linear group* $GL(n,\mathbb{Z})$, the set of all invertible $n\times n$ matrices over $\mathbb{Z}$.

Elements of $GL(n,\mathbb{Z})$ are also called *unimodular matrices*. Unimodular matrices are important objects in the theory of lattices because they classify lattices up to change of basis: two bases $B$ and $C$ generate the same lattice $L$ if and only if they are related by $B = CU$ for some unimodular matrix $U$.

The problem of randomly generating unimodular matrices in an 'unbiased' way therefore arises when one needs to select a random lattice. This is a problem of significant importance. One notable application is in lattice-based cryptography, where the strength of a cryptographic protocol is derived from the difficulty of certain problems concerning lattices. For example, this is an essential part of a lattice-based cryptosystem, where the random generation of high-dimensional unimodular matrices (usually over finite fields) takes the role of the random generation of large primes in RSA. Ensuring high quality of the random lattice generation algorithm is essential to avoid leaving the cryptosystem vulnerable to various attacks. Indeed, a number of algorithms exist for randomly selecting a unimodular matrix, but in terms of statistical properties they are not exactly equivalent, leading to empirical disparities in the resilience of their associated cryptosystems. Besides cryptography, random unimodular matrices also see application in the study of lattices and lattice algorithms in general.

## Random elements of $SL(2,\mathbb{Z})$

The focus of this repository, however, is just on a simple toy problem in low dimensions: the generation of random $2\times 2$ integer matrices of unit determinant, i.e. elements of $SL(2,\mathbb{Z})$. Even in such low dimensions, this problem admits some interesting algorithmic complications.

The natural first obstacle is that $SL(2,\mathbb{Z})$ is an infinite group, and admits no uniform probability distribution. Some arbitrary choices need to be made to define a notion of an 'unbiased' sampling of $SL(2,\mathbb{Z})$, and it is not immediately evident that any two sampling algorithms will be statistically equivalent.

We first address the naive approach to the sampling algorithm, sample-and-replace:

* Sample a $2\times 2$ matrix $M$ uniformly at random from the hypercube $[-N,N]^4$.
* If $\det M = 1$, keep; otherwise, discard $M$ and sample again until an element of $SL(2,\mathbb{Z})$ is found.

Although in theory every element of $SL(2,\mathbb{Z}) \cap [-N,N]^4$ is equally likely to be selected, the practicality of this algorithm is nil if we need to generate many diverse samples. This is because the number of elements of $SL(n,\mathbb{Z})$ of Frobenius norm $\leq N$ grows as $\sim N^{n^2-n}$. Taking $n=2$ and aliasing $[-N,N]^4$ for the ball in $SL(2,\mathbb{Z})$ of radius $N$ under the Frobenius norm, we thus expect the density of $SL(2,\mathbb{Z})$ in the ball to decay as $\sim N^{-2}$. Thus the group $SL(2,\mathbb{Z})$ is quite sparse in the space of all matrices. Even for matrices with entries bounded by merely $100$, we would expect to need to sample on the order of $10^4$ matrices just to generate a single keepable element.

Therefore in terms of practicality we are restricted to algorithms that are guaranteed to generate elements of $SL(2,\mathbb{Z})$ either always or with high probability. This is a problem that has seen some research efforts, and indeed some deterministic algorithms are available in various software packages including Sage and Magma, though I am not well-read on the statistical properties of these algorithms.

## A GAN-based approach

In this repository we experiment with an option that, as far as I am aware, has not been explored previously. As a quick summary, the method works as follows:

* Generate a dataset consisting of a large number of previously generated elements of $SL(2,\mathbb{Z})$.
* Train a generative adversarial network to produce new elements of $SL(2,\mathbb{Z})$ by using the dataset as training data for the discriminator.

In particular, I would like to see if it is possible to train the GAN on a dataset generated by a biased random matrix generator without the GAN inheriting its biases as well.

The random matrix generator I have in mind uses the fact that $SL(2,\mathbb{Z})$ is generated by two elements:

$$
S = \begin{pmatrix}
0 & -1\\
1 & 0
\end{pmatrix},
~ T = \begin{pmatrix}
1 & 1\\
0 & 1
\end{pmatrix}.
$$
In addition, $S^2 = -I$ while $(ST)^3 = -I$; in particular $SL(2,\mathbb{Z})$ is also generated by $S$ and $ST$, which both have finite orders $4$ and $6$ respectively. Hence $SL(2,\mathbb{Z})$ is isomorphic to the free product $\mathbb{Z}/4\mathbb{Z} * \mathbb{Z}/6\mathbb{Z}$. We can therefore represent each element of $SL(2,\mathbb{Z})$ as a vertex in the Cayley graph of $\mathbb{Z}/4\mathbb{Z} * \mathbb{Z}/6\mathbb{Z}$, and randomly sample elements in this Cayley graph via random walks. This suggests the following sampling algorithm:

* Generate uniformly at random a number $\ell$ in the range $0,\ldots,N$.
* Perform a random walk in $SL(2,\mathbb{Z})$ of length $\ell$ starting at the origin, where at each step we right-multiply by either $S$ or $ST$ with equal probability.
* Return the endpoint.

This algorithm is guaranteed to generate elements of $SL(2,\mathbb{Z})$. However, it is a biased algorithm: the samples are not expected to equidistribute in the ball of radius $N$ within the Cayley graph. In fact, if the behavior here is anything similar to the standard random walk on $\mathbb{Z}^2$, then they are not even expected to follow a Gaussian distribution. On the other hand, a large number of samples far from the identity can be computed somewhat efficiently. Therefore it would be interesting to see if a GAN can learn from a dataset generated by this random sampling algorithm without picking up on the biased distribution.

# Implementation and outcomes

A relativistic GAN architecture with gradient penalty was selected after some amount of experimentation with other architectures such as WGAN-GP. To generate the training dataset, I wrote a simple program that generates random walks in $SL(2,\mathbb{Z})$ (see `matrix_methods.py`). The program can efficiently generate on the order of 1e7 unimodular matrices from random walks of length 1e3.

For both the generator and discriminator, a simple fully connected network was selected with explicit encoding of polynomial features of degree 2. The determinant of a $2\times 2$ matrix is a quadratic polynomial in the entries, and it seemed appropriate to allow the network to use these features directly rather than force it to learn a simple, already known polynomial relationship from scratch.

Raw training of the GAN architecture on the matrix data did not appear to result in learning. It was clear that the GAN could not learn the condition that all entries must be integers. However, it also did not appear to be able to learn the determinantal condition. I changed the goalposts slightly and encoded the integrality and determinant conditions explicitly as loss functions for the generator, with the hope that the network would use the training data to encourage diverse outputs mimicking the training data distribution.

Unfortunately, although the network was now capable of generating floating-point matrices in $SL(2,\mathbb{R})$ (which, by continuity, would generically round to matrices in $SL(2,\mathbb{Z})$, it was now left with the mode collapse problem and would only generate matrices similar to a single element. Some experimentation was performed to see if this issue could be resolved via architectural choices, including the implementation of PAC-GAN, but to no avail.

It is unclear to me what would be necessary to resolve the mode collapse problem. The discriminator architecture should in theory be sufficient to produce a perfect discriminator, given that the polynomial relationship given by membership in $SL(2,\mathbb{R})$ can be exactly recovered. The generator architecture could probably allow for more parameters (at the moment it is a very small network) in order to construct a more feature-rich loss landscape. It is also quite likely that a different strategy for generating the training data could be useful: perhaps the random walk algorithm is somehow producing a training dataset where the discriminator finds it difficult to learn the determinantal relationship, and instead learns some extraneous feature corrupting the generator's learning ability. For now I will shelve this project, but may return to it another time with other approaches.