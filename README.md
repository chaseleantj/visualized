# Visualized

A tool that animates 2D linear transformations.

<img src="resources/demo.jpg">

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [How to use](#how-to-use)
* [Credits](#credits)
* [Notes](#notes)

## General info

This program animates linear transformations. First, you initialize the vectors and matrices. Then, you can perform the transformation to see how the matrix acts on those vectors. By entering any matrix of your choice, you can scale, shear, rotate, reflect and project the 2D plane. 

It can also demonstrate how determinants, eigenvectors and the null space of a matrix describe a linear transformation.
	
## Setup

Download and run `Visualized.exe`

## How to Use

Click the video below to view.

<a href="https://www.youtube.com/watch?v=XZR3rKZ1UTM" target="_blank"><img src="https://img.youtube.com/vi/XZR3rKZ1UTM/maxresdefault.jpg" width=50%></a>

In the demonstration above, the composite matrix transformation $M_{3}\cdot M_{2}\cdot M_{1}$ is applied to the vectors $\vec{v_{1}}$ and $\vec{v_{2}}$.

Click <a href="Files/Icons/instructions.png">here</a> for the full instructions. You can also access it by clicking **Help** when running `Visualized.exe`.


## Credits

* This program was heavily inspired by 3Blue1Brown's Youtube series, <a href="https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab">Essence of Linear Algebra</a>. He has undoubtedly inspired many generations of students to give math a little more love.
* The background music is <a href="https://www.fesliyanstudios.com/royalty-free-music/download/elven-forest/376">Elven Forest</a> by David Renda.

## Notes

Linear interpolation was used to animate a linear transformation. This causes issues with transformations that *describe* rotations such as 

$$ 
\begin{bmatrix}
\cos(\pi) & -\sin(\pi)\\
\sin(\pi) & \cos(\pi)
\end{bmatrix}
$$

The animation for the above transformation will look like a reflection instead of a rotation. I have manually written code to detect rotations and animate them as such, but a better approach would be to use quaternions, which I have yet to implement.

Also, panning by clicking and dragging and zooming with the mouse wheel is not supported. I have enabled it in a later project, Flow Field, which aims to visualize vector fields. If you liked this program, you'll enjoy Flow Field. You can find it here.
