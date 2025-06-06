#!/usr/bin/env nemesis

import gmsh
from pylith.meshio.gmsh_utils import (BoundaryGroup, MaterialGroup, GenerateMesh)

class App(GenerateMesh):
    """
    Block is DOMAIN_X by DOMAIN_Y with discretization size DX.

    p4------p6------p3
    |       |        |
    |       |        |
    |       |        |
    |       |        |
    p1------p5------p2
    """
    DOMAIN_X = DOMAIN_Y = 8.0e+3
    DX = 1.0e+3

    def __init__(self):
        super().__init__()
        self.cell_choices = {
            "required": True,
            "choices": ["tri", "quad"],
            }

    def create_geometry(self):
        """Create geometry.
        """
        lx = self.DOMAIN_X
        ly = self.DOMAIN_Y
        x0 = -0.5 * lx
        y0 = -0.5 * ly

        p1 = gmsh.model.geo.add_point(x0, y0, 0.0)
        p2 = gmsh.model.geo.add_point(x0+lx, y0, 0.0)
        p3 = gmsh.model.geo.add_point(x0+lx, y0+ly, 0.0)
        p4 = gmsh.model.geo.add_point(x0, y0+ly, 0.0)

        p5 = gmsh.model.geo.add_point(x0+0.5*lx, y0, 0.0)
        p6 = gmsh.model.geo.add_point(x0+0.5*lx, y0+ly, 0.0)

        self.l_yneg0 = gmsh.model.geo.add_line(p1, p5)
        self.l_yneg1 = gmsh.model.geo.add_line(p5, p2)
        self.l_xpos = gmsh.model.geo.add_line(p2, p3)
        self.l_ypos1 = gmsh.model.geo.add_line(p3, p6)
        self.l_ypos0 = gmsh.model.geo.add_line(p6, p4)
        self.l_xneg = gmsh.model.geo.add_line(p4, p1)
        self.l_fault = gmsh.model.geo.add_line(p5, p6)

        c0 = gmsh.model.geo.add_curve_loop([self.l_yneg0, self.l_fault, self.l_ypos0, self.l_xneg])
        self.s_xneg = gmsh.model.geo.add_plane_surface([c0])
        c1 = gmsh.model.geo.add_curve_loop([self.l_yneg1, self.l_xpos, self.l_ypos1, -self.l_fault])
        self.s_xpos = gmsh.model.geo.add_plane_surface([c1])

        gmsh.model.geo.synchronize()

    def mark(self):
        """Mark geometry for materials, boundary conditions, faults, etc.
        """
        materials = (
            MaterialGroup(tag=1, entities=[self.s_xneg]),
            MaterialGroup(tag=2, entities=[self.s_xpos]),
        )
        for material in materials:
            material.create_physical_group()

        face_groups = (
            BoundaryGroup(name="boundary_xneg", tag=10, dim=1, entities=[self.l_xneg]),
            BoundaryGroup(name="boundary_xpos", tag=11, dim=1, entities=[self.l_xpos]),
            BoundaryGroup(name="boundary_yneg", tag=12, dim=1, entities=[self.l_yneg0, self.l_yneg1]),
            BoundaryGroup(name="boundary_ypos", tag=13, dim=1, entities=[self.l_ypos0, self.l_ypos1]),
        )
        for group in face_groups:
            group.create_physical_group()

        patch_groups = (
            BoundaryGroup(name="patch_xneg", tag=20, dim=2, entities=[self.s_xneg]),
            BoundaryGroup(name="patch_xpos", tag=21, dim=2, entities=[self.s_xpos]),
        )
        for group in patch_groups:
            group.create_physical_group(recursive=True)

    def generate_mesh(self, cell):
        """Generate the mesh. Should also include optimizing the mesh quality.
        """
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.DX)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.DX)
        if cell == "quad":
            gmsh.model.mesh.set_transfinite_automatic(recombine=True)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 8)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")


if __name__ == "__main__":
    App().main()


# End of file
