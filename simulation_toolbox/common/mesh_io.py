#!/usr/bin/env python3
"""
Readers for CARP mesh files, IGB simulation output, and ParaView camera files.

These functions are vendored from two libraries by M. Strocchi that are not
publicly available, so that this repository can be cloned and run without them:

  - read_pts, read_elem, read_IGB_file, read_mesh, carp_to_pyvista
        from GSA_library.mesh_utils
  - read_pvcc_paraview_file
        from GSA_library.pyvista_utils
  - read_tets
        from SIMULATION_library.mesh_utils

Only the functions used by this repository are included. The behaviour is
unchanged; the indentation has been normalised to 4 spaces.
"""

import os
import re
import sys

import numpy as np
import pyvista as pv
import vtk


def read_pts(filename):
    """Read a CARP .pts file into an (n_points, 3) array."""
    print('Reading ' + filename + '...')
    return np.loadtxt(filename, dtype=float, skiprows=1)


def read_elem(filename, el_type='Tt', tags=True):
    """Read a CARP .elem file, optionally keeping the region tag column."""
    print('Reading ' + filename + '...')

    if el_type == 'Tt':
        if tags:
            return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1, 2, 3, 4, 5))
        else:
            return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1, 2, 3, 4))
    elif el_type == 'Tr':
        if tags:
            return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1, 2, 3, 4))
        else:
            return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1, 2, 3))
    elif el_type == 'Ln':
        if tags:
            return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1, 2, 3))
        else:
            return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1, 2))
    else:
        raise Exception('element type not recognised. Accepted: Tt, Tr, Ln')


def read_tets(elemname):
    """Read the tetrahedra (with tags) of a CARP .elem file."""
    return np.loadtxt(elemname, dtype=int, usecols=[1, 2, 3, 4, 5], skiprows=1)


def read_IGB_file(igbfname: str):
    """Read a CARP .igb file, returning (header dict, data array)."""
    header_size = 256
    parsed_header = {}
    try:
        with open(igbfname, 'rb') as f:
            header = f.read(header_size)
        header = header.decode("utf-8")
        for jj in header.strip().split():
            [key, val] = jj.split(':')
            if val.isdigit():
                parsed_header[key] = int(val)
            else:
                parsed_header[key] = val

        # Now read the data and create an array
        with open(igbfname, 'rb') as f:
            y = np.fromfile(f, 'f4')
        y = y[header_size:]
        nt = parsed_header['t']
        nx = parsed_header['x']
        nentries = y.shape[0]

        ntot = nt * nx
        if parsed_header['type'] == 'vec3f':
            ntot *= 3
            if nentries == ntot:
                y = np.reshape(y, (nt, nx, 3))

        else:
            if nentries == ntot:
                y = np.reshape(y, (nt, nx))
            elif nentries > ntot:
                print('Warning: discarding the last {0} elements (problems in igb file)'.format(nentries - ntot))
                y = y[:ntot]
            else:  # nentries < ntot
                nt = nentries // nx
                if nt == 0:
                    print('ERROR: y too short! ({0} elements; expected {1} (problems in igb file)'.format(nentries, ntot))
                    sys.exit()
                else:
                    ntot = nt * nx
                    y = y[:ntot]
                    print('Warning: missing {0} elements to reach {1}(problems in igb file); reshaping to {2} time steps'.format(nentries % nx, parsed_header['t'], nt))
                    parsed_header['t'] = nt

            y = np.reshape(y, (nt, nx))

        return parsed_header, y
    except ValueError:
        print('error with {0}'.format(igbfname))


def read_mesh(meshname):
    """Read a CARP mesh, converting from binary with meshtool if needed."""
    if not os.path.exists(meshname + ".elem"):

        cmd = ["meshtool convert", "-imsh=" + meshname, "-omsh=" + meshname,
               "-ifmt=carp_bin", "-ofmt=carp_txt"]
        cmd_str = ' '.join(cmd)
        os.system(cmd_str)

        pts = read_pts(meshname + ".pts")
        elem = read_elem(meshname + ".elem", el_type='Tt', tags=True)

        os.system("rm " + meshname + ".pts")
        os.system("rm " + meshname + ".elem")
        os.system("rm " + meshname + ".lon")

    else:

        pts = read_pts(meshname + ".pts")
        elem = read_elem(meshname + ".elem", el_type='Tt', tags=True)

    return pts, elem


def carp_to_pyvista(pts, elem):
    """Build a pyvista UnstructuredGrid of tetrahedra from points and elements."""
    if elem.shape[1] == 5:
        elem = elem[:, :4]

    tets = np.column_stack((np.ones((elem.shape[0],), dtype=int) * 4, elem)).flatten()
    cell_type = np.ones((elem.shape[0],), dtype=int) * vtk.VTK_TETRA

    plt_msh = pv.UnstructuredGrid(tets, cell_type, pts)

    return plt_msh


def read_pvcc_paraview_file(pvcc_file):
    """Parse a ParaView .pvcc camera file into a dict of camera settings."""
    file_content = open(pvcc_file, 'r')
    lines = file_content.read().splitlines()

    dct = {}

    properties = ["CameraPosition", "CameraFocalPoint", "CameraViewUp", "CameraViewAngle"]
    labels = ["position", "focal_point", "up", "view_angle"]

    for i in range(len(properties)):
        record = False
        values = []
        for j in range(len(lines)):
            line = lines[j]

            if '<Property name="' + properties[i] + '"' in line:
                record = True

            if record:
                pattern = re.search('value="(.*?)"/>', line)
                if pattern is not None:
                    values.append(float((pattern.group(1))))

            if '</Property>' in line:
                record = False
                dct[labels[i]] = values

    dct["view_angle"] = dct["view_angle"][0]

    return dct
