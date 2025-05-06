(sec-user-physics-boundary-conditions)=
# Boundary Conditions

## Assigning boundary conditions

There are three basic steps in assigning a specific boundary condition to a portion of the domain.

1. Create sets of vertices in the mesh generation process for each boundary condition.
2. Set the parameters for each boundary condition group using `cfg` files or command line arguments.
3. Specify the spatial variation in parameters for the boundary condition using a spatial database.

## Marking boundaries when meshing

The procedure for marking boundaries depends on the mesh generator.
For meshes specified using the PyLith mesh ASCII format, cell faces are specified using `face-group`s (see {ref}`sec-user-file-formats-meshio-ascii`).
In Cubit cell faces are marked using sidesets.
Note also that we currently associate boundary conditions with string identifiers.
Finally, note that the boundary conditions must be associated with a simply-connected surface;
that is, surfaces must be connected and not contain holes.

## Arrays of boundary condition components

A dynamic array of boundary condition components associates a name (string) with each boundary condition. The default boundary condition for each component in the array is `DirichletTimeDependent`.
Other boundary conditions can be bound to the named items in the array by assining the component type to the named boundary condition.

```{code-block} cfg
---
caption: Array of boundary conditions in a `cfg` file
---
[pylithapp.problem]
# Array of four boundary conditions
bc = [x_neg, x_pos, y_pos, z_neg]
# Default boundary condition is DirichletBC
# Keep default value for x_neg and x_pos but assign new types to y_pos and z_neg
bc.y_pos = pylith.bc.AbsorbingDampers
bc.z_neg = pylith.bc.NeumannTimeDependent
```

## Diagnostic information

The diagnostic information includes the outward normal direction (`normal_dir`) and the two tangential directions (`horizontal_tangential_dir` and `vertical_tangential_dir`).
The default basis order for discretizing these directions is 1, so these produce `vertex_fields` as opposed to `cell_fields` (basis order of 0).

## Boundary condition implementations

:::{toctree}
time-dependent.md
absorbing-dampers.md
:::

:::{seealso}
See {ref}`sec-user-governing-eqns` for the derivation of the finite-element formulation for each of the boundary conditions.
:::
