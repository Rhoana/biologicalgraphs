// standard includes
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

// andres graph includes
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"


unsigned char *CppMulticut(long nvertices, long nedges, long *vertex_ones, long *vertex_twos, double *edge_weights, double beta)
{
    // create the empty graph structure
    andres::graph::Graph<> graph;
    std::vector<double> weights(nedges);

    // add in all of the vertices
    graph.insertVertices(nvertices);

    // populate the edges
    for (long ie = 0; ie < nedges; ++ie) {
        graph.insertEdge(vertex_ones[ie], vertex_twos[ie]);

        // a low beta value encouranges not merging - note the edge_weights are probability of merging
        // compared to the original greedy-additive algorithm which was probablity of boundary
        weights[ie] = log(edge_weights[ie] / (1.0 - edge_weights[ie])) + log((1.0 - beta) / beta);
    }

    // create empty edge labels and call the kernighan-lin algorithm
    std::vector<char> edge_labels(nedges, 1);
    
    andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);
    
    // turn vector into char array and return
    unsigned char *maintained_edges = new unsigned char[nedges];
    for (long ie = 0; ie < nedges; ++ie) {
        maintained_edges[ie] = edge_labels[ie];
    }

    return maintained_edges;
}
