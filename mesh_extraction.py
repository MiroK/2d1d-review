from dolfin import Mesh, MeshEditor, SubsetIterator, near, Cell, Edge
from dolfin import EdgeFunction, MeshFunction
from collections import defaultdict, OrderedDict, deque
from random import sample
import numpy as np


def analyse_connectivity(mesh, edge_f, subdomain):
    '''
    Compute connectivity between vertices of mesh and its edges that are marked
    by edge function with value of subdomain.
    '''
    # We are only interested in gdim == tdim meshes
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()
    assert gdim == tdim and tdim > 1
   
    # And building 1d in gdim meshes
    try:
        edim = edge_f.dim()
        assert edim == 1
        edges = SubsetIterator(edge_f, subdomain)

    except AttributeError:
        assert isinstance(edge_f, list)
        # Recreate edges from list
        mesh.init(1)
        edges = (Edge(mesh, index) for index in edge_f)

    # First compute connectivity edge <--> vertex for edge, vertex on edge_f
    mesh.init(1, 0)
    topology = mesh.topology()
    # Connect between all edge->c of mesh
    all_edge_vertex_c = topology(1, 0)   

    # Connectivity edge->vertex on interval mesh
    # Edge function - extract marked
    # if isinstance(edge_f, MeshFunction):
    edge_vertex_c = dict((edge.index(), set(all_edge_vertex_c(edge.index())))
                         for edge in edges)
        # Only marked are given in list
    # else:
    #     edge_vertex_c = dict((edge, set(all_edge_vertex_c(edge)))
    #                          for edge in edge_f)

    assert len(edge_vertex_c), 'No edges with value %d' % subdomain

    # Connectivity vertex->edge on interval mesh
    vertex_edge_c = defaultdict(list)
    for edge, vertices in edge_vertex_c.iteritems():
        for vertex in vertices:
            vertex_edge_c[vertex].append(edge)

    return edge_vertex_c, vertex_edge_c


def add_cells(start, cells, edges, edge_vertex_c, vertex_edge_c):
    '''
    Traverse graph=(edge_vertex_c, vertex_edge_c) from start and add cells
    from global vertices as you go. These will be cells of interval mesh and
    each one is mapped one-to-one to some edge of the mesh.
    '''
    # If I can travel from start over som edge go ahead
    if vertex_edge_c[start]:
        # Remove edge that will get me furher
        edge = vertex_edge_c[start].pop()
        # Further is the other vertex of this edge
        vertex = edge_vertex_c[edge].difference({start}).pop()
        # Once there, the new vertex can forget how I came
        vertex_edge_c[vertex].remove(edge)

        # Create the cell from global vertex indices
        cells.append([start, vertex])
        # The cell is edge of the mesh
        edges.append(edge)

        # Bifurcations are points of recursion starting from it
        for n in range(len(vertex_edge_c[vertex])):
            add_cells(vertex, cells, edges, edge_vertex_c, vertex_edge_c)
    # Exhaused all and can return
    else:
        return 1


def find_branches(cells, singles, bifurcations, branches):
    '''
    A branch by our definition is formed by cells which connect two vertices
    each of which is either a bifurcation or vertex connected to single edge. 
    '''
    branch = []
    # Add cells to branch until you hit a bifurcation or single
    while cells:
        cell = cells.popleft()
        branch.append(cell)
        # When that happens start a new branch
        if cell[1] in singles or cell[1] in bifurcations:
            branches.append(branch)
            find_branches(cells, singles, bifurcations, branches)

    return 1


def build_interval_mesh(vertices, cells):
    '''Mesh of 1d topology in n-dims.'''
    imesh = Mesh()
    editor = MeshEditor()
    editor.open(imesh, 1, vertices.shape[1])
    editor.init_vertices(len(vertices))
    editor.init_cells(len(cells))

    # Add vertices
    for vertex_index, v in enumerate(vertices):
        editor.add_vertex(vertex_index, v)

    # Add cells
    for cell_index, (v0, v1) in enumerate(cells):
        editor.add_cell(cell_index, v0, v1)

    editor.close()

    return imesh


def interval_mesh_from_edge_f(mesh, edge_f, subdomain, single_imesh=True):
    '''
    Build a mesh describing domain of topological dim 1 in the d-dim space.
    The new mesh consists edges of mesh where the edge function takes
    value of subdomain. Build a map from cells of new mesh to edges of mesh.
    '''
    edge_vertex_c, vertex_edge_c = analyse_connectivity(mesh, edge_f, subdomain)

    # Find vertices that are connected to only one edge. Traversing starts there
    # typically
    begin_end = set(vertex 
                    for vertex in vertex_edge_c
                    if len(vertex_edge_c[vertex]) == 1)
    # Now bifurcations
    bifurcations = set(vertex
                       for vertex in vertex_edge_c if
                       len(vertex_edge_c[vertex]) > 2)

    # Cells will be formed by lists of length two describing vertices of the
    # global mesh
    cells = []
    # Each cell of interval mesh is mapped to one edge of mesh
    edges = []

    # Chose where to start
    # Closed loop
    if not begin_end:
        # Loops with bifurcations should start there
        if bifurcations:
            start = sample(bifurcations, 1)[0]
        # Without the bifurcation all the vertices are equal
        else:
            start = vertex_edge_c.keys()[0]
    # There is a at least one ending
    else:
        n_bifurcations = len(bifurcations)
        # Sanity check for single 'straight' beams
        if len(begin_end) == 2:
            assert n_bifurcations == 0
        # In different case there must be atleast one bifurcation, otherwise we
        # have two or more strucutures that better be considered separetely
        else:
            msg = 'Non-intersecting structures cannot be handled as single interval mesh.'
            assert n_bifurcations > 0, msg

        start = begin_end.pop()

    # Traverse
    add_cells(start, cells, edges, edge_vertex_c, vertex_edge_c)
    
    # In the future it might be useful to handle bifurcated meshes as a
    # collection of simple/straight meshes which I reffer to as branch.
    # By default the structure is a single interval mesh = single branch
    if single_imesh:
        branches = [cells]
    # Otherwise the branch spanes from single-edge-vertex/bifurcation to
    # another single-edge-vertex/bifurcation
    else:
        branches = []
        begin_end.add(start)
        cells = deque(cells)
        find_branches(cells, begin_end, bifurcations, branches)

    # Each branch refers to mesh(global) vertices in its own local numbering and
    # build its separate mappping from local cell indices to mesh(global) edges.
    len_branches = map(len, branches)
    branch_offsets = [0]
    [branch_offsets.append(branch_offsets[-1]+n) for n in len_branches]
    # edges[branch_offsets[i]:branch_offsets[i+1]] are the edges of i-th branch

    # Get vertices that are needed to create interval mesh
    gdim = mesh.geometry().dim()
    vertices = mesh.coordinates().reshape((-1, gdim))
    # Collect interval meshes and their maps
    imeshes = []
    edge_maps = []
    for i, cells in enumerate(branches):
        # We now build a map from global vertices of mesh to local vertices in
        # interval mesh. Ordered dict add keys only once and remembers their
        # order, i.e. the local index 
        gl_map = OrderedDict.fromkeys(sum(cells, []))
        # Assign to each global vertex its local vertex
        for l_vertex, g_vertex in enumerate(gl_map):
            gl_map[g_vertex] = l_vertex

        # Translate cells that make up branch to local numbering of vertices
        cells = map(lambda pair: (gl_map[pair[0]], gl_map[pair[1]]), cells)
        
        # Get vertices that make up this branch
        branch_vertices = vertices[gl_map.keys(), :]

        # Build the interval mesh from its cells and vertices
        interval_mesh = build_interval_mesh(branch_vertices, cells)
        edge_map = edges[branch_offsets[i]:branch_offsets[i+1]]
        assert interval_mesh.num_cells() == len(edge_map)

        imeshes.append(interval_mesh)
        edge_maps.append(edge_map)

    return imeshes, edge_maps
