/*
  Compressed sparse row graph structure for use with boost::graph.

  Copyright 2012–2013 Daniel Müllner, http://math.stanford.edu/~muellner

  This file is derived from the C++ header file
      compressed_sparse_row_graph.hpp
  in the boost::graph library. The original authors are Jeremiah Willcock,
  Douglas Gregor and Andrew Lumsdaine.

  This file implements a compressed sparse row graph type for use with
  boost::graph. In contrast to the above mentioned, very generic boost header
  file, the present file requires a very specific input data structure. The
  data arrays are not copied in the constructor, so that the structure is very
  efficient, provided that the data has already the correct structure.

  The data structure is as follows:  The array "edgelist" contains the targets
  of all edges. The array "weightlist" contains the edge weights. The edges
  are sorted by their source vertex. Edges rowstart[0] to rowstart[1]-1 start
  at vertex 0, edges rowstart[1] to rowstart[2]-1 start at vertex 1, and so on.
  The following data is given to the contructor:

    num_vertices, num_edges : number of vertices and edges, of course.
    edgelist                : array of size (num_edges)
    weightlist              : array of size (num_edges)
    rowstart                : array of size (num_vertices+1)

  This file is part of the Python Mapper package, an open source tool
  for exploration, analysis and visualization of data. Python Mapper is
  distributed under the GPLv3 license. See the project home page

    http://math.stanford.edu/~muellner/mapper

  for more information.
*/

#ifndef CSR_GRAPH_HPP
#define CSR_GRAPH_HPP

// optional: do a concept check at the end
//#define CSR_GRAPH_CONCEPT_CHECK
#include <boost/config.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/iterator/counting_iterator.hpp>


namespace csr_graph {

  /* Edge descriptor

     An edge descriptor is a pair (vertex, index) of integers, where vertex
     is the start vertex of the edge and index is the total index in
     [0,...(num_edges-1)].
  */
  template<typename Graph>
  class csr_edge_descriptor
  {
  private:
    typedef typename Graph::vertex_descriptor vertex_descriptor;
    typedef typename Graph::edges_size_type edges_size_type;
    typedef typename Graph::out_edge_iterator out_edge_iterator;

  public:
    vertex_descriptor vertex;
    edges_size_type idx;

    csr_edge_descriptor() // only needed for concept checking
      : vertex(0)
      , idx(0)
    {}

    csr_edge_descriptor(vertex_descriptor const vertex_, edges_size_type const idx_)
      : vertex(vertex_)
      , idx(idx_)
    {}

    inline bool operator==(const csr_edge_descriptor& e) const {
      return idx == e.idx;}
    inline bool operator!=(const csr_edge_descriptor& e) const {
      return idx != e.idx;}
    inline bool operator<(const csr_edge_descriptor& e) const {
      return idx < e.idx;}
    inline bool operator>(const csr_edge_descriptor& e) const {
      return idx > e.idx;}
    inline bool operator<=(const csr_edge_descriptor& e) const {
      return idx <= e.idx;}
    inline bool operator>=(const csr_edge_descriptor& e) const {
      return idx >= e.idx;}
  };

  /*
     Iterator over all edges in the graph.
  */
  template<typename Graph>
  class csr_edge_iterator
  {
  public:
    typedef typename Graph::edge_descriptor value_type;

  private:
    typedef typename Graph::edges_size_type edges_size_type;
    typedef typename Graph::vertex_descriptor vertex_descriptor;

    value_type current_edge;
    edges_size_type end_of_this_vertex;
    edges_size_type total_num_edges;
    edges_size_type const * rowstart;

  public:
    csr_edge_iterator()
      : current_edge(0,0)
      , end_of_this_vertex(0)
      , total_num_edges(0)
    {}

    csr_edge_iterator(Graph const & graph,
                      vertex_descriptor const vertex,
                      edges_size_type const idx,
                      edges_size_type const end_of_this_vertex_)
      : current_edge(vertex, idx),
        end_of_this_vertex(end_of_this_vertex_),
        total_num_edges(num_edges(graph)),
        rowstart(graph.rowstart)
    {}

    // Gives an error message in the concept check:
    //typedef boost::bidirectional_traversal_tag iterator_category;
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef std::ptrdiff_t difference_type;
    typedef value_type const * pointer;
    typedef value_type const & reference;

    reference operator *() const {return current_edge;}

    inline bool operator ==(csr_edge_iterator const & i) const {
      return current_edge==i.current_edge;
    }

    inline bool operator !=(csr_edge_iterator const & i) const {
      return current_edge!=i.current_edge;
    }

    csr_edge_iterator & operator ++() {
      ++current_edge.idx;
      while (current_edge.idx==end_of_this_vertex &&
             current_edge.idx!=total_num_edges) {
        ++current_edge.vertex;
        end_of_this_vertex = rowstart[current_edge.vertex + 1];
      }
      return *this;
    }

    csr_edge_iterator & operator --() {
      if (current_edge.idx==0) {
        throw 0;
        //--current_edge.idx;
        //current_edge.vertex = -1;
        //end_of_this_vertex = rowstart[0];
      }
      else {
        --current_edge.idx;
        while (current_edge.idx<rowstart[current_edge.vertex]) {
          --current_edge.vertex;
          end_of_this_vertex = rowstart[current_edge.vertex + 1];
        }
      }
      return *this;
    }

    csr_edge_iterator operator ++(int) {
      csr_edge_iterator const ret(*this);
      ++(*this);
      return ret;
    }

    csr_edge_iterator operator --(int) {
      csr_edge_iterator const ret(*this);
      --(*this);
      return ret;
    }
  };

  /*
     Iterator over all edges incident to a single vertex
  */
  template<typename Graph>
  class csr_out_edge_iterator
  {
  public:
    typedef typename Graph::edge_descriptor value_type;

  private:
    typedef typename Graph::edges_size_type edges_size_type;
    typedef typename Graph::vertex_descriptor vertex_descriptor;

    value_type current_edge;

   public:
    csr_out_edge_iterator()
      : current_edge(0,0)
    {}

    csr_out_edge_iterator(vertex_descriptor const vertex,
                          edges_size_type const idx)
      : current_edge(vertex,idx)
    {}

    // Gives an error message in the concept check:
    //typedef boost::random_access_iterator_tag iterator_category;
    typedef std::random_access_iterator_tag iterator_category;
    typedef std::ptrdiff_t difference_type;
    typedef value_type const * pointer;
    typedef value_type const & reference;

    inline reference operator *() const {return current_edge;}

    inline bool operator ==(csr_out_edge_iterator const & i) const {
      return current_edge==i.current_edge;
    }

    inline bool operator !=(csr_out_edge_iterator const & i) const {
      return current_edge!=i.current_edge;
    }

    inline csr_out_edge_iterator & operator ++() {
      ++current_edge.idx;
      return *this;
    }

    inline csr_out_edge_iterator & operator --() {
      --current_edge.idx;
      return *this;
    }
    csr_out_edge_iterator operator ++(int) {
      csr_out_edge_iterator const ret(*this);
      ++(*this);
      return ret;
    }
    csr_out_edge_iterator operator --(int) {
      csr_out_edge_iterator const ret(*this);
      --(*this);
      return ret;
    }
    inline void operator +=(difference_type const n) {
      current_edge.idx += static_cast<edges_size_type>(n);
    }
    inline void operator -=(difference_type const n) {
      current_edge.idx -= static_cast<edges_size_type>(n);
    }
    csr_out_edge_iterator operator +(difference_type const n) const {
      csr_out_edge_iterator ret(*this);
      ret += n;
      return ret;
    }
    csr_out_edge_iterator operator -(difference_type const n) const {
      csr_out_edge_iterator ret(*this);
      ret -= n;
      return ret;
    }
    inline difference_type operator -(csr_out_edge_iterator const & i) const {
      return current_edge.idx-(*i).idx;
    }
    inline bool operator <(csr_out_edge_iterator const & i) const {
      return this->current_edge.idx<i.current_edge.idx;
    }
    inline bool operator >(csr_out_edge_iterator const & i) const {
      return this->current_edge.idx>i.current_edge.idx;
    }
    inline bool operator >=(csr_out_edge_iterator const & i) const {
      return this->current_edge.idx>=i.current_edge.idx;
    }
    inline bool operator <=(csr_out_edge_iterator const & i) const {
      return this->current_edge.idx<=i.current_edge.idx;
    }
    value_type operator [](difference_type const n) const {
      value_type return_edge(current_edge);
      return_edge.idx += static_cast<edges_size_type>(n);
      return return_edge;
    }
  };

  template<typename Graph>
  inline csr_out_edge_iterator<Graph> operator +
  (
   ptrdiff_t const n,
   csr_out_edge_iterator<Graph> const & i
   ) {
    return i+n;
  }

  /* Main class: Compressed sparse row graph.

     Vertex and EdgeIndex should be unsigned integral types.

  */

  template<typename Vertex, typename EdgeIndex, typename Weight>
  class csr_graph
  {
  public:
    // not needed for boost, but for my own templated classes
    typedef Weight weight_type;

    typedef boost::no_property vertex_property_type;

    // Concept requirements:
    // For Graph
    typedef Vertex vertex_descriptor;
    typedef csr_edge_descriptor<csr_graph> edge_descriptor;
    typedef boost::directed_tag directed_category;
    typedef boost::allow_parallel_edge_tag edge_parallel_category;

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#endif
    struct traversal_category: public boost::vertex_list_graph_tag,
                               public boost::edge_list_graph_tag,
                               public boost::incidence_graph_tag
    {};
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    // For VertexListGraph
    typedef boost::counting_iterator<Vertex> vertex_iterator;
    typedef Vertex vertices_size_type;

    // For EdgeListGraph
    typedef csr_edge_iterator<csr_graph> edge_iterator;
    typedef EdgeIndex edges_size_type;

    // For IncidenceGraph
    typedef csr_out_edge_iterator<csr_graph> out_edge_iterator;
    typedef EdgeIndex degree_size_type;

    // For AdjacencyGraph
    typedef Vertex const * adjacency_iterator;

    // For BidirectionalGraph (not implemented)
    typedef void in_edge_iterator;

  public:
    vertex_descriptor const * edgelist;
  public:
    edges_size_type const * rowstart;
    edges_size_type const num_edges;
    vertices_size_type const num_vertices;

    //  Constructor
    csr_graph(Vertex const * const edgelist_,
              EdgeIndex const * const rowstart_,
              EdgeIndex const num_edges_,
              Vertex const num_vertices_)
      : edgelist(edgelist_)
      , rowstart(rowstart_)
      , num_edges(num_edges_)
      , num_vertices(num_vertices_)
    {}

    inline static Vertex null_vertex() {
      /* This should never be needed. However, an algorithm might use a null_vertex some time, and it does not
       * necessarily indicate an error. In this case, fix it by choosing an appropriate value. For the time being,
       * I am lazy and do not invent a "null" value.
       */
      throw 0;
    }
  };

  // From VertexListGraph
  template<typename Graph>
  inline typename Graph::vertex_descriptor
  num_vertices(Graph const & g) {
    return g.num_vertices;
  }

  template<typename Graph>
  std::pair<typename Graph::vertex_iterator,
            typename Graph::vertex_iterator>
  inline vertices(Graph const & g) {
    return std::make_pair(typename Graph::vertex_iterator(0),
                          typename Graph::vertex_iterator(g.num_vertices));
  }

  // From AdjacencyGraph
  template<typename Graph>
  inline std::pair<typename Graph::adjacency_iterator,
                   typename Graph::adjacency_iterator>
  adjacent_vertices(typename Graph::vertex_descriptor const v,
                    Graph const & g)
  {
    typename Graph::adjacency_iterator start = &g.edgelist[g.rowstart[v]];
    typename Graph::adjacency_iterator end   = &g.edgelist[g.rowstart[v+1]];
    return std::make_pair(start, end);
  }

  // From IncidenceGraph
  template<typename Graph>
  inline typename Graph::vertex_descriptor
  source(typename Graph::edge_descriptor const e,
         Graph const &)
  {
    return e.vertex;
  }

  template<typename Graph>
  inline typename Graph::vertex_descriptor
  target(typename Graph::edge_descriptor const e,
         Graph const & g)
  {
    return g.edgelist[e.idx];
  }

  template<typename Graph>
  std::pair<typename Graph::out_edge_iterator,
            typename Graph::out_edge_iterator>
  out_edges(typename Graph::vertex_descriptor const v,
         Graph const & g)
  {
    typename Graph::out_edge_iterator const start(v, g.rowstart[v]);
    typename Graph::out_edge_iterator const end(v, g.rowstart[v + 1]);
    return std::make_pair(start,end);
  }

  template<typename Graph>
  inline typename Graph::degree_size_type
  out_degree(typename Graph::vertex_descriptor const v,
         Graph const & g)
  {
    return g.rowstart[v+1]-g.rowstart[v];
  }

  template<typename Graph>
  inline typename Graph::edges_size_type
  num_edges(Graph const & g)
  {
    return g.num_edges;
  }

  template<typename Graph>
  std::pair<typename Graph::edge_iterator,
            typename Graph::edge_iterator>
  edges(Graph const & g)
  {
    typedef typename Graph::edge_iterator edge_iterator;
    typedef typename Graph::vertex_descriptor vertex_descriptor;
    if (g.num_edges==0) {
      return std::make_pair(edge_iterator(), edge_iterator());
    } else {
      // Find the first vertex that has outgoing edges
      vertex_descriptor src = 0;
      while (g.rowstart[src+1] == 0) ++src;
      return std::make_pair(edge_iterator(g, src, 0, g.rowstart[src + 1]),
                            edge_iterator(g, g.num_vertices, g.num_edges, 0));
    }
  }

  template<typename Graph>
  inline boost::identity_property_map
  get(boost::vertex_index_t, Graph const &)
  {
    return boost::identity_property_map();
  }

  /*
     Weight map: indexed by edges, read-only
  */

  template<typename Graph>
  class csr_weight_map {
  public:
    typedef boost::readable_property_map_tag category;
    typedef typename Graph::weight_type value_type;
    typedef value_type const & reference;
    typedef typename Graph::edge_descriptor key_type;

    csr_weight_map(value_type const * const weight_map_)
      : weight_map(weight_map_)
    {}

    inline reference operator [](key_type const i) const {
      return weight_map[i.idx];
    }

  private:
    value_type const * weight_map;
  };

  template<typename Graph>
  inline typename csr_weight_map<Graph>::value_type
  get(csr_weight_map<Graph> const map,
      typename csr_weight_map<Graph>::key_type const key) {
    return map[key];
  }

  /*
     Distance map: indexed by vertices, read-write
  */

  template<typename Graph>
  class csr_distance_map {
  public:
    typedef boost::read_write_property_map_tag category;
    typedef typename Graph::weight_type value_type;
    typedef value_type & reference;
    typedef typename Graph::vertex_descriptor key_type;

    csr_distance_map(value_type * const distance_map_)
      : distance_map(distance_map_)
    {}

    inline reference operator [](key_type const i) const {
      return distance_map[i];
    }

  private:
    value_type * distance_map;
  };

  template<typename Graph>
  inline typename csr_distance_map<Graph>::value_type
  get(csr_distance_map<Graph> const map,
      typename csr_distance_map<Graph>::key_type const key) {
    return map[key];
  }

  template<typename Graph>
  inline void
  put(csr_distance_map<Graph> const map,
      typename csr_distance_map<Graph>::key_type const key,
      typename csr_distance_map<Graph>::value_type const val) {
    map[key] = val;
  }

  /*
     Color map: indexed by vertices, read-write
  */

  template<typename Graph>
  class csr_color_map {
  public:
    typedef boost::read_write_property_map_tag category;
    typedef boost::default_color_type value_type;
    typedef value_type & reference;
    typedef typename Graph::vertex_descriptor key_type;

    csr_color_map(value_type * const color_map_)
      : color_map(color_map_)
    {}

    inline reference operator [](key_type const i) const {
      return color_map[i];
    }

  private:
    value_type * color_map;
  };

  template<typename Graph>
  inline typename csr_color_map<Graph>::value_type
  get(csr_color_map<Graph> const map,
      typename csr_color_map<Graph>::key_type const key) {
    return map[key];
  }

  template<typename Graph>
  inline void
  put(csr_color_map<Graph> const map,
      typename csr_color_map<Graph>::key_type const key,
      typename csr_color_map<Graph>::value_type const val) {
    map[key] = val;
  }

  /*
     Predecessor map: indexed by vertices, read-write
  */

  template<typename Graph>
  class csr_predecessor_map {
  public:
    typedef boost::read_write_property_map_tag category;
    typedef typename Graph::vertex_descriptor value_type;
    typedef value_type & reference;
    typedef typename Graph::vertex_descriptor key_type;

    csr_predecessor_map(value_type * const predecessor_map_)
      : predecessor_map(predecessor_map_)
    {}

    inline reference operator [](key_type const i) const {
      return predecessor_map[i];
    }

  private:
    value_type * predecessor_map;
  };

  template<typename Graph>
  inline typename csr_predecessor_map<Graph>::value_type
  get(csr_predecessor_map<Graph> const map,
      typename csr_predecessor_map<Graph>::key_type const key) {
    return map[key];
  }

  template<typename Graph>
  inline void
  put(csr_predecessor_map<Graph> const map,
      typename csr_predecessor_map<Graph>::key_type const key,
      typename csr_predecessor_map<Graph>::value_type const val) {
    map[key] = val;
  }

  /*
     Dummy map: indexed by vertices, write only
  */

  template<typename Graph>
  class csr_dummy_map {
  public:
    typedef boost::writable_property_map_tag category;
    typedef typename Graph::vertex_descriptor value_type;
    // Somehow, a property map needs a reference type, even
    // if it's read-only (Boost's property_map.hpp, line 33)
    typedef value_type & reference;
    typedef typename Graph::vertex_descriptor key_type;
  };

  template<typename Graph>
  inline void
  put(csr_dummy_map<Graph>,
      typename csr_dummy_map<Graph>::key_type,
      typename csr_dummy_map<Graph>::value_type)
  { }

} // end namespace csr_graph

#ifdef CSR_GRAPH_CONCEPT_CHECK
using namespace boost;

typedef csr_graph::csr_graph<unsigned int, unsigned int, double> Graph;

BOOST_CONCEPT_ASSERT((VertexListGraphConcept<Graph>));
BOOST_CONCEPT_ASSERT((EdgeListGraphConcept<Graph>));
BOOST_CONCEPT_ASSERT((IncidenceGraphConcept<Graph>));

BOOST_CONCEPT_ASSERT((BidirectionalIteratorConcept<Graph::edge_iterator>));
BOOST_CONCEPT_ASSERT((RandomAccessIteratorConcept<Graph::out_edge_iterator>));
BOOST_CONCEPT_ASSERT((RandomAccessIteratorConcept<Graph::vertex_iterator>));
BOOST_CONCEPT_ASSERT((RandomAccessIteratorConcept<Graph::adjacency_iterator>));
#undef CSR_GRAPH_CONCEPT_CHECK
#endif // CSR_GRAPH_CONCEPT_CHECK

#endif // CSR_GRAPH_HPP
