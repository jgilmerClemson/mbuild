__all__ = ['Sc', 'Bcc', 'Fcc', 'Diamond']

import numpy as np
from mbuild.lattice import Lattice


def spacing_error_cubic(lattice_spacings):
    """Determine if the inputted lattice spacings are the proper size.

    spacing_error_cubic raises a more verbose exception if the user passes
    in a value for the lattice spacings that is not a singlular value that
    can be converted to a float.

    Parameters:
    -----------
    lattice_spacings: float, required
        Single value defining the edge lengths of the cubic cell.
    """

    try:
        float(lattice_spacings)
    except TypeError:
        raise TypeError('Too many lattice_spacings for type cubic. '
                        'Spacings {} were passed in, expected 1 value that can'
                        ' be converted to a float.'.format(lattice_spacings))


def expand_spacings_3d_cubic(lattice_spacings):
    """Convert single lattice spacing to list of lattice spacings.

    expand_spacings_3d converts the singluar input for the length of the
    cubic lattice to a list of 3. These represent a,b,c which are all
    equivalent in the case of the cubic lattices.

    Parameters:
    -----------
    lattice_spacings: float, required
        Single value defining the edge lengths of the cubic cell.
    """

    lattice_spacings = float(lattice_spacings)
    return [lattice_spacings for x in range(3)]


def compound_dict_expansion(basis_atoms, compound_dict, num_basis_atoms):
    """If user only passes in one basis atom, all basis atoms are same type.

    compound_dict_expansion allows the user to either fully define their
    system with unique compounds for each sample, or if only one compound is
    passed in, assume homogenous crystal.

    Parameters:
    -----------
    basis_atoms: dict, required
    Dictionary relating lattice basis positions to an identifier.

    compound_dict: dict, required
    Dictionary relating Compound to same identifier from basis_atoms dict.

    num_basis_atoms: int, required, >= 1
    Total number of unique basis atoms in unit cell.
    """

    if isinstance(compound_dict, dict):
        if len(compound_dict.keys()) == num_basis_atoms:
            pass
        elif len(compound_dict.keys()) == 1:
            for key in basis_atoms.keys():
                try:
                    compound_dict[key] = compound_dict['A']
                except KeyError:
                    raise KeyError('Invalid key within compound_dict. \'A\' '
                                   'is the expcted id.')
        else:
            raise ValueError('Incorrect amount of keys provided in '
                             'compound_dict. Expected {} or 1, recieved {}.'
                             .format(num_basis_atoms,
                                     len(compound_dict.keys())))
    else:
        raise TypeError('Invalid type for compound_dict: type {} '
                        'was passed, dict expected.'
                        .format(type(compound_dict)))
    return compound_dict


class Sc(Lattice):
    """Simple cubic Bravais lattice system.

    Attributes
    ----------
    dimension : int, default=3
        Dimension of the system of interest.
    lattice_vectors : numpy array, shape=(3,3), optional
                      default=([1,0,0], [0,1,0], [0,0,1])
        Vectors that define edges of the cubic system.
    lattice_spacings : float, shape=1, required, default=None
        Length of unit cell edge.
    lattice_points : dictionary, shape={'id':[list of atom positions]},
    default=[0, 0, 0]
        Location of all basis compounds in unit cell.
    angles : list-like,  shape=(dimension,), optional, default=None
        Interplanar angles describing unit cell.

    Simple Cubic Defaults
    ---------------------
    dimension : 3
    lattice_vectors : [1, 0, 0
                       0, 1, 0
                       0, 0, 1]

    basis_atoms : 'A', [[0, 0, 0]]
    """

    dimension = 3
    lattice_vectors = np.identity(dimension, dtype=float)
    basis_atoms = {'A': [[0, 0, 0]]}

    def __init__(self, lattice_spacings):
        spacing_error_cubic(lattice_spacings=lattice_spacings)

        super().__init__(lattice_spacings=expand_spacings_3d_cubic(lattice_spacings),
                         dimension=self.dimension,
                         basis_atoms=self.basis_atoms,
                         lattice_vectors=self.lattice_vectors
                         )

    def populate(self, compound_dict=None, x=1, y=1, z=1):
        compound_dict = compound_dict_expansion(self.basis_atoms,
                                                compound_dict, 1)
        return super().populate(compound_dict=compound_dict,
                                x=x,
                                y=y,
                                z=z)


class Bcc(Lattice):
    """Body Centered cubic Bravais lattice system.

    Attributes
    ----------
    dimension : int, default=3
        Dimension of the system of interest.
    lattice_vectors : numpy array, shape=(dimension, dimension), optional
                      default=([1,0,0], [0,1,0], [0,0,1])
        Vectors that define edges of the cubic system.
    lattice_spacings : float, shape=1, required, default=None
        Length of unit cell edges.
    basis_atoms : dictionary, shape={'id':[list of atom positions]},
    default=Bcc basis positions
        Location of all basis compounds in unit cell.
    angles : list-like,  shape=(dimension,), optional, default=None
        Interplanar angles describing unit cell.

    Simple Cubic Defaults
    ---------------------
    dimension : 3
    lattice_vectors : [1, 0, 0
                       0, 1, 0
                       0, 0, 1]

    basis_atoms : 'A' : [0, 0, 0], 'B' : [0.5, 0.5, 0.5]
    """

    dimension = 3
    lattice_vectors = np.identity(dimension, dtype=float)
    basis_atoms = {'A': [[0, 0, 0]],
                   'B': [[0.5, 0.5, 0.5]]}

    def __init__(self, lattice_spacings):
        spacing_error_cubic(lattice_spacings=lattice_spacings)
        super().__init__(lattice_spacings=expand_spacings_3d_cubic(lattice_spacings),
                         dimension=self.dimension,
                         basis_atoms=self.basis_atoms,
                         lattice_vectors=self.lattice_vectors)

    def populate(self, compound_dict=None, x=1, y=1, z=1):
        compound_dict = compound_dict_expansion(self.basis_atoms,
                                                compound_dict, 2)
        return super().populate(compound_dict=compound_dict,
                                x=x,
                                y=y,
                                z=z)


class Fcc(Lattice):
    """Face Centered Bravais lattice system.

    Attributes
    ----------
    dimension : int, default=3
        Dimension of the system of interest.
    lattice_vectors : numpy array, shape=(dimension, dimension), optional
                      default=([1,0,0], [0,1,0], [0,0,1])
        Vectors that define edges of the cubic system.
    lattice_spacings : float, shape=1, required, default=None
        Length of unit cell edges.
    basis_atoms : dictionary, shape={'id':[list of atom positions]},
    default=Fcc basis positions
        Location of all basis compounds in unit cell.
    angles : list-like,  shape=(dimension,), optional, default=None
        Interplanar angles describing unit cell.

    Simple Cubic Defaults
    ---------------------
    dimension : 3
    lattice_vectors : [1, 0, 0
                       0, 1, 0
                       0, 0, 1]

    basis_atoms = {'A': [[0, 0, 0]],
                   'B': [[0, 0.5, 0.5]],
                   'C': [[0.5, 0, 0.5]],
                   'D': [[0.5, 0.5, 0]]}
    """

    dimension = 3
    lattice_vectors = np.identity(dimension, dtype=float)
    basis_atoms = {'A': [[0, 0, 0]],
                   'B': [[0, 0.5, 0.5]],
                   'C': [[0.5, 0, 0.5]],
                   'D': [[0.5, 0.5, 0]]}

    def __init__(self, lattice_spacings):
        spacing_error_cubic(lattice_spacings=lattice_spacings)
        super().__init__(lattice_spacings=expand_spacings_3d_cubic(lattice_spacings),
                         dimension=self.dimension,
                         basis_atoms=self.basis_atoms,
                         lattice_vectors=self.lattice_vectors)

    def populate(self, compound_dict=None, x=1, y=1, z=1):
        compound_dict = compound_dict_expansion(self.basis_atoms,
                                                compound_dict, 4)
        return super().populate(compound_dict=compound_dict,
                                x=x,
                                y=y,
                                z=z)


class Diamond(Lattice):
    """Diamond lattice lattice system.

    Attributes
    ----------
    dimension : int, default=3
        Dimension of the system of interest.
    lattice_vectors : numpy array, shape=(dimension, dimension), optional
                      default=([1,0,0], [0,1,0], [0,0,1])
        Vectors that define edges of the diamond system.
    lattice_spacings : float, shape=1, required, default=None
        Length of unit cell edges.
    basis_atoms : dictionary, shape={'id':[list of atom positions]},
    default=Diamond basis positions
        Location of all basis compounds in unit cell.
    angles : list-like,  shape=(dimension,), optional, default=None
        Interplanar angles describing unit cell.

    Simple Cubic Defaults
    ---------------------
    dimension : 3
    lattice_vectors : [1, 0, 0
                       0, 1, 0
                       0, 0, 1]

    basis_atoms = {'A': [[0, 0, 0]],
                   'B': [[0, 0.5, 0.5]],
                   'C': [[0.5, 0, 0.5]],
                   'D': [[0.5, 0.5, 0]]
                   'E': [[0.25, 0.25, 0.75]]
                   'F': [[0.75, 0.25, 0.25]]
                   'G': [[0.25, 0.75, 0.25]]
                   'H': [[0.75, 0.75, 0.75]]}
    """

    dimension = 3
    lattice_vectors = np.identity(dimension, dtype=float)
    basis_atoms = {'A': [[0, 0, 0]],
                   'B': [[0, 0.5, 0.5]],
                   'C': [[0.5, 0, 0.5]],
                   'D': [[0.5, 0.5, 0]],
                   'E': [[0.25, 0.25, 0.75]],
                   'F': [[0.75, 0.25, 0.25]],
                   'G': [[0.25, 0.75, 0.25]],
                   'H': [[0.75, 0.75, 0.75]]}

    def __init__(self, lattice_spacings):
        spacing_error_cubic(lattice_spacings=lattice_spacings)
        super().__init__(lattice_spacings=expand_spacings_3d_cubic(lattice_spacings),
                         dimension=self.dimension,
                         basis_atoms=self.basis_atoms,
                         lattice_vectors=self.lattice_vectors)

    def populate(self, compound_dict=None, x=1, y=1, z=1):
        compound_dict = compound_dict_expansion(self.basis_atoms,
                                                compound_dict, 8)
        return super().populate(compound_dict=compound_dict,
                                x=x,
                                y=y,
                                z=z)
