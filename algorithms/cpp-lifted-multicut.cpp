// standard includes
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>

// andres graph includes
#include "andres/graph/graph.hxx"
#include "andres/graph/complete-graph.hxx"
#include "andres/graph/multicut-lifted/greedy-additive.hxx"



unsigned char *CppLiftedMulticut(long nvertices, long nedges, long *vertex_ones, long *vertex_twos, double *lifted_weights, double beta)
{
    // create the empty graph structure
    andres::graph::Graph<> original_graph(nvertices);

    // insert edges for all of the adjacent vertices
    for (long ie = 0; ie < nedges; ++ie) {
        original_graph.insertEdge(vertex_ones[ie], vertex_twos[ie]);
    }

    // create the lifted graph
    andres::graph::CompleteGraph<> lifted_graph(nvertices); 
    std::vector<double> weights(lifted_graph.numberOfEdges());

    // create all of the lifted edge weights
    for (long iv1 = 0; iv1 < nvertices; ++iv1) {
        for (long iv2 = iv1 + 1; iv2 < nvertices; ++iv2) {
            // update the lifted weight between these two vertices
            double probability = lifted_weights[iv1 * nvertices + iv2];

            // multiply these weights by the number of normal weights to balance out effects
            weights[lifted_graph.findEdge(iv1, iv2).second] = nedges * (log(probability / (1.0 - probability)) + log((1.0 - beta) / beta));
        }
    }
    // populate all of the normal weights with their regular values
    for (long ie = 0; ie < nedges; ++ie) {
        long vertex_one = vertex_ones[ie];
        long vertex_two = vertex_twos[ie];
        double probability = lifted_weights[vertex_one * nvertices + vertex_two];

        // multiply these weights by the number of lifted weights to balance out effects
        weights[lifted_graph.findEdge(vertex_one, vertex_two).second] = nvertices * (nvertices - 1) / 2 * (log(probability / (1.0 - probability)) + log((1.0 - beta) / beta));
    }

    // create empty edge labels
    std::vector<char> edge_labels(lifted_graph.numberOfEdges(), 1);

    andres::graph::multicut_lifted::greedyAdditiveEdgeContraction(original_graph, lifted_graph, weights, edge_labels);

    // turn vector into char array and return
    unsigned char *maintained_edges = new unsigned char[nedges];
    for (long ie = 0; ie < nedges; ++ie) {
        maintained_edges[ie] = edge_labels[lifted_graph.findEdge(vertex_ones[ie], vertex_twos[ie]).second];
    }

    return maintained_edges;
}
