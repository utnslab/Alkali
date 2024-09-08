
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include <algorithm>
#include <fstream>
#include <bitset>
#include <string>
#include <random>
#include <queue>

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <boost/config.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/filtered_graph.hpp>

#include "ep2/passes/LiftUtils.h"
#include "ep2/passes/Mapping.h"

#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const& e){
  llvm::errs() << e.what() << '\n';
  abort();
}

using namespace mlir;

namespace mlir {
namespace ep2 {

typedef void* vertex_t;
typedef size_t weight_t;
static constexpr long long INF_WT = 1e18;
static constexpr long long INF_MIN = 1e17;
static constexpr long long SMALL_WT = 1;
static constexpr size_t N_RAND_ITERS = 1000;
static constexpr int MIN_CUT_SUCCESS = 0;
static constexpr int MIN_CUT_FAILURE_INF_FLOW = 1;
static constexpr int MIN_CUT_FAILURE_SRC_INF_CAP_PATH = 2;
static constexpr int MIN_CUT_FAILURE_SINK_INF_CAP_PATH = 3;
static constexpr int MIN_CUT_FAILURE_NOT_PIPELINE = 4;
static const char* errMsgs[4] = {"",
                                 "Inf flow returned by max_flow solver.",
                                 "In collapsing src+n, src->n would create inf capacity path.",
                                 "In collapsing sink+n, n->sink would create inf capacity path."};


template <typename weight_map_t>
struct dfs_visitor_t : public boost::default_dfs_visitor {
  dfs_visitor_t(weight_map_t w_map, std::unordered_map<size_t, int>* weights) : w_map_(w_map), weights_(weights) {}

  template <typename vertex, typename graph>
  void discover_vertex(vertex u, const graph& g) {
    put(w_map_, u, weights_ ? (*weights_)[u] : 1);
  }

  weight_map_t w_map_;
  std::unordered_map<size_t, int>* weights_;
};

struct dfs_path_visitor_t : public boost::default_dfs_visitor {
  dfs_path_visitor_t(std::vector<size_t>& stack, size_t sink, bool& found_sink) : stack_(stack), sink_(sink), found_sink_(found_sink) {}

  template <typename vertex, typename graph>
  void discover_vertex(vertex u, const graph& g) {
    if (!found_sink_) {
      stack_.push_back(u);
      if (u == sink_) {
        found_sink_ = true;
      }
    }
  }

  template <typename vertex, typename graph>
  void finish_vertex(vertex u, const graph& g) {
    // preserve the path if found_sink_ is true.
    if (!found_sink_) {
      assert(stack_.back() == u);
      stack_.pop_back();
    }
  }

  std::vector<size_t>& stack_;
  size_t sink_;
  bool& found_sink_;
};


template <typename capacity_map_t>
struct inf_edge_capacity {
  inf_edge_capacity() {}
  inf_edge_capacity(capacity_map_t c_map) : c_map_(c_map) {}
  template <typename edge>
  bool operator()(const edge& e) const {
    return get(c_map_, e) == INF_WT;
  }
  capacity_map_t c_map_;
};

template <typename EdgeWeightMap>
struct positive_edge_weight {
  positive_edge_weight() { }
  positive_edge_weight(EdgeWeightMap weight) : m_weight(weight) { }
  template <typename Edge>
  bool operator()(const Edge& e) const {
    return 0 < get(m_weight, e);
  }
  EdgeWeightMap m_weight;
};

static std::string opToId(mlir::Operation* op) {
  if (isa<ep2::BitCastOp>(op)) {
    return "bc";
  } else if (isa<ep2::ConstantOp>(op)) {
    return "k";
  } else if (isa<ep2::AddOp>(op)) {
    return "add";
  } else if (isa<ep2::SubOp>(op)) {
    return "sub";
  } else if (isa<ep2::BitSetOp>(op)) {
    return "bs";
  } else if (isa<ep2::BitGetOp>(op)) {
    return "bg";
  } else if (isa<ep2::CmpOp>(op)) {
    return "?";
  } else if (isa<ep2::CallOp>(op)) {
    return "fx";
  } else if (isa<ep2::ContextRefOp>(op)) {
    return "ctxR";
  } else if (isa<ep2::ExtractOp>(op)) {
    return "x";
  } else if (isa<ep2::ExtractValueOp>(op)) {
    return "xv";
  } else if (isa<ep2::ExtractOffsetOp>(op)) {
    return "xf";
  } else if (isa<ep2::EmitOp>(op)) {
    return "m";
  } else if (isa<ep2::EmitValueOp>(op)) {
    return "mv";
  } else if (isa<ep2::EmitOffsetOp>(op)) {
    return "mf";
  } else if (isa<ep2::LookupOp>(op)) {
    return "tl";
  } else if (isa<ep2::UpdateOp>(op)) {
    return "tu";
  } else if (isa<ep2::LoadOp>(op)) {
    return "ld";
  } else if (isa<ep2::StoreOp>(op)) {
    return "st";
  } else if (isa<ep2::NopOp>(op)) {
    return "nop";
  } else if (isa<ep2::GlobalOp>(op)) {
    return "glb";
  } else if (isa<ep2::GlobalImportOp>(op)) {
    return "imp";
  } else if (isa<ep2::InitOp>(op)) {
    return "ini";
  } else if (isa<ep2::MulOp>(op)) {
    return "mul";
  } else if (isa<ep2::TerminateOp>(op)) {
    return "ter";
  } else if (isa<ep2::ReturnOp>(op)) {
    return "ret";
  } else if (isa<ep2::StructAccessOp>(op)) {
    return "sac";
  } else if (isa<ep2::StructUpdateOp>(op)) {
    return "sup";
  } else if (isa<ep2::StructConstantOp>(op)) {
    return "sk";
  } else if (isa<arith::ConstantOp>(op)) {
    return "sk";
  } else if (isa<arith::AddIOp>(op)) {
    return "addi";
  } else if (isa<arith::CmpIOp>(op)) {
    return "cmpi";
  } else {
    return "";
  }
}

// accept myAdjList by value, so we can freely manipulate it here.
static int runBalancedMinCut(
  // Graph information
  std::unordered_map<vertex_t, std::unordered_map<vertex_t, long long>>
      myAdjList,
  vertex_t source, vertex_t sink,

  float tol, double sourceWeight, double &resultSourceWeight,
  std::unordered_set<vertex_t> &sourceSet,
  std::vector<std::pair<std::pair<vertex_t, vertex_t>, long long>>
      &falseEdges,
  std::unordered_map<vertex_t, int> &v_weights,
  std::vector<std::pair<vertex_t, vertex_t>> &pipelineConstraints,
  std::unordered_map<vertex_t, std::string>& vtxNames
  ) {

  typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::bidirectionalS> Traits;
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
    boost::property<boost::vertex_index_t, size_t>,
    boost::property<boost::edge_capacity_t, long,
      boost::property<boost::edge_residual_capacity_t, long,
        boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>> Graph;
  typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<Graph>::edge_descriptor edge_descriptor;

  // Every step, re-construct the graph.
  while (true) {
    Graph g;
    Graph g_base;
    typedef typename boost::property_map<Graph, boost::edge_capacity_t>::type capacity_map_t;
    capacity_map_t capacity = get(boost::edge_capacity, g);
    boost::property_map<Graph, boost::edge_residual_capacity_t>::type residual_capacity =
      get(boost::edge_residual_capacity, g);

    boost::property_map<Graph, boost::edge_reverse_t>::type rev = get(boost::edge_reverse, g);

    std::vector<vertex_descriptor> verts;
    std::vector<vertex_descriptor> verts_base;
    std::unordered_map<vertex_t, size_t> vtxToVecIdx;
    std::unordered_map<vertex_descriptor, vertex_t> boostToMyVtx;

    std::vector<vertex_t> iterOrder;
    for (const auto& ve : myAdjList) {
      iterOrder.push_back(ve.first);
    }
    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::shuffle(iterOrder.begin(), iterOrder.end(), rand_gen);
    for (vertex_t v : iterOrder) {
      vtxToVecIdx[v] = verts.size();
      verts.push_back(boost::add_vertex(g));
      verts_base.push_back(boost::add_vertex(g_base));
      boostToMyVtx[verts.back()] = v;
    }

    auto addEdge = [&](vertex_t src, vertex_t dst, long long wt, bool addToBase) {
      edge_descriptor e1, e2;
      bool in1, in2;

      boost::tie(e1, in1) = boost::add_edge(verts[vtxToVecIdx[src]], verts[vtxToVecIdx[dst]], g);
      boost::tie(e2, in2) = boost::add_edge(verts[vtxToVecIdx[dst]], verts[vtxToVecIdx[src]], g);
      assert(in1 && in2);

      if (addToBase) {
        edge_descriptor e3;
        bool in3;
        boost::tie(e3, in3) = boost::add_edge(verts_base[vtxToVecIdx[src]], verts_base[vtxToVecIdx[dst]], g_base);

        assert(in3);
        capacity[e3] = wt;
      }

      capacity[e1] = wt;
      capacity[e2] = 0;
      rev[e1] = e2;
      rev[e2] = e1;
    };

    for (const auto& pr : myAdjList) {
      vertex_t src = pr.first;
      for (const auto& pr2 : pr.second) {
        vertex_t dst = pr2.first;
        long long wt = pr2.second;
        assert(wt > 0);
        addEdge(src, dst, wt, true);
      }
    }

    for (const auto& edge : falseEdges) {
      vertex_t s = edge.first.first;
      vertex_t e = edge.first.second;
      long long wt = edge.second;
      addEdge(s, e, wt, false);
    }

    {
      inf_edge_capacity<capacity_map_t> filter(capacity);
      boost::filtered_graph<Graph, inf_edge_capacity<capacity_map_t>> inf_graph(g, filter);

      std::vector<size_t> stack;
      bool found_sink = false;
      dfs_path_visitor_t vis(stack, verts[vtxToVecIdx[sink]], found_sink);
      auto index_map = boost::get(boost::vertex_index, inf_graph);
      auto color_map = boost::make_vector_property_map<boost::default_color_type>(index_map);
      boost::depth_first_visit(inf_graph, verts[vtxToVecIdx[source]], vis, color_map);
      if (found_sink) {
        llvm::errs() << "dfs stack: ";
        for (size_t v : stack) {
          llvm::errs() << v << ',';
        }
        llvm::errs() << '\n';
      }
    }

    std::vector<vertex_descriptor> topo_sort;
    boost::topological_sort(g_base, std::back_inserter(topo_sort));
    std::unordered_map<vertex_descriptor, size_t> topo_sort_pos;
    for (size_t i = 0; i<topo_sort.size(); ++i) {
      topo_sort_pos[topo_sort[i]] = i;
    }

    long flow = boost::push_relabel_max_flow(g, verts[vtxToVecIdx[source]], verts[vtxToVecIdx[sink]]);
    llvm::errs() << "FLOW: " << flow << '\n';
    assert(flow >= 0);

    auto dumpGraphViz = [&]() {
      // https://dreampuf.github.io/GraphvizOnline
      // Apparently boost's graphviz api requires rtti, which is somehow disabled in this project?
      boost::graph_traits<Graph>::vertex_iterator u_iter, u_end;
      boost::graph_traits<Graph>::out_edge_iterator ei, e_end;
      llvm::errs() << "source -> v" << verts[vtxToVecIdx[source]] << ";\n";
      for (boost::tie(u_iter, u_end) = boost::vertices(g); u_iter != u_end; ++u_iter) {
        for (boost::tie(ei, e_end) = boost::out_edges(*u_iter, g); ei != e_end; ++ei) {
          llvm::errs() << "v" << *u_iter << " -> " << "v" << boost::target(*ei, g) << " [ label = " << (capacity[*ei] == INF_WT ? "INF" : std::to_string(capacity[*ei])) << " color = " << (residual_capacity[*ei] == 0 ? "red" :  "black") << " ];\n";
        }
      }
      llvm::errs() << "v" << verts[vtxToVecIdx[sink]] << " -> sink;\n";
    };

    boost::remove_edge_if([&](const edge_descriptor& e){
      return residual_capacity[e] == 0;
    }, g);

    std::vector<long long> wmap(verts.size());
    typedef boost::iterator_property_map<std::vector<long long>::iterator, boost::property_map<Graph, boost::vertex_index_t>::const_type> g_weight_map_t;
    g_weight_map_t g_wmap(wmap.begin(), get(boost::vertex_index, g));
    std::unordered_map<size_t, int> vertex_weights;
    for (const auto& pr : v_weights) {
      vertex_weights[vtxToVecIdx[pr.first]] = pr.second;
    }
    dfs_visitor_t<g_weight_map_t> vis(g_wmap, &vertex_weights);

    // dfs from source in the residual graph.
    auto index_map = boost::get(boost::vertex_index, g);
    auto color_map = boost::make_vector_property_map<boost::default_color_type>(index_map);
    boost::depth_first_visit(g, verts[vtxToVecIdx[source]], vis, color_map);

    auto dumpGraphBaseViz = [&]() {
      // https://dreampuf.github.io/GraphvizOnline
      // Apparently boost's graphviz api requires rtti, which is somehow disabled in this project?
      boost::graph_traits<Graph>::vertex_iterator u_iter, u_end;
      boost::graph_traits<Graph>::out_edge_iterator ei, e_end;
      for (boost::tie(u_iter, u_end) = boost::vertices(g_base); u_iter != u_end; ++u_iter) {
        llvm::errs() << "v" << (*u_iter) << " [ color = " << (wmap[*u_iter] > 0 ? "red" : "blue") << " ]\n";
      }
      llvm::errs() << "source -> v" << verts[vtxToVecIdx[source]] << ";\n";
      for (boost::tie(u_iter, u_end) = boost::vertices(g_base); u_iter != u_end; ++u_iter) {
        for (boost::tie(ei, e_end) = boost::out_edges(*u_iter, g_base); ei != e_end; ++ei) {
          llvm::errs() << "v" << *u_iter << " -> " << "v" << boost::target(*ei, g_base) << " [ label = " << (capacity[*ei] == INF_WT ? "INF" : std::to_string(capacity[*ei])) << " color = " << (boost::edge(boost::source(*ei, g_base), boost::target(*ei, g_base), g).second ? "black" : "green") << " ];\n";
        }
      }
      llvm::errs() << "v" << verts[vtxToVecIdx[sink]] << " -> sink;\n";
    };

    if (flow >= INF_MIN) {
      return MIN_CUT_FAILURE_INF_FLOW;
    }

    {
      for (const auto& e : pipelineConstraints) {
        if (wmap[vtxToVecIdx[e.first]] == 0 && wmap[vtxToVecIdx[e.second]] > 0) {
          llvm::errs() << "NOT PIPELINE_ABLE\n";
          llvm::errs() << verts[vtxToVecIdx[e.first]] << " " << verts[vtxToVecIdx[e.second]] << '\n';
          auto orig_vertices = boost::vertices(g_base);
          for (auto vit = orig_vertices.first; vit != orig_vertices.second; ++vit) {
            llvm::errs() << "NAME " << (*vit) << " " << vtxNames[boostToMyVtx[*vit]] << '\n';
          }
          dumpGraphViz();
          dumpGraphBaseViz();
          return MIN_CUT_FAILURE_NOT_PIPELINE;
        }
      }
    }

    size_t wmap_sum = 0;
    for (size_t i = 0; i<verts.size(); ++i) {
      wmap_sum += wmap[i];
    }

    // find inf-capacity paths from source, sink in g.
    inf_edge_capacity<capacity_map_t> filter(capacity);
    boost::filtered_graph<Graph, inf_edge_capacity<capacity_map_t>> inf_graph(g, filter);
    auto inf_graph_rev = boost::make_reverse_graph(inf_graph);

    std::vector<bool> inf_path_to_source(verts.size());
    {
      typedef boost::iterator_property_map<std::vector<bool>::iterator, boost::property_map<Graph, boost::vertex_index_t>::const_type> vis_map_t;
      vis_map_t vis_map(inf_path_to_source.begin(), get(boost::vertex_index, inf_graph));
      dfs_visitor_t<vis_map_t> vis(vis_map, nullptr);
      auto index_map = boost::get(boost::vertex_index, inf_graph);
      auto color_map = boost::make_vector_property_map<boost::default_color_type>(index_map);
      boost::depth_first_visit(inf_graph, verts[vtxToVecIdx[source]], vis, color_map);
    }
    std::vector<bool> inf_path_to_sink(verts.size());
    {
      typedef boost::iterator_property_map<std::vector<bool>::iterator, boost::property_map<Graph, boost::vertex_index_t>::const_type> vis_map_t;
      vis_map_t vis_map(inf_path_to_sink.begin(), get(boost::vertex_index, inf_graph_rev));
      dfs_visitor_t<vis_map_t> vis(vis_map, nullptr);
      auto index_map = boost::get(boost::vertex_index, inf_graph_rev);
      auto color_map = boost::make_vector_property_map<boost::default_color_type>(index_map);
      boost::depth_first_visit(inf_graph_rev, verts[vtxToVecIdx[sink]], vis, color_map);
    }

    float frac_src_cut = ((float) wmap_sum) / verts.size();
    llvm::errs() << "FRAC: " << llvm::format("%.2f", frac_src_cut) << " " << llvm::format("%.2f", sourceWeight*(1-tol)) << " " << llvm::format("%.2f", sourceWeight*(1+tol)) << '\n';

    resultSourceWeight = frac_src_cut;

    /*
    Iterate over all edges in g_base, where wmap[src] != 0 and wmap[dst] == 0.
    These are the cut edges. Now we want to add the topologically first such dst into our set.
    */
    auto orig_edges = boost::edges(g_base);
    if (frac_src_cut < sourceWeight * (1-tol)) {
      //llvm::errs() << "collapse source\n";
      ssize_t earliest = topo_sort_pos.size();
      vertex_descriptor chosen;

      for (auto eit = orig_edges.first; eit != orig_edges.second; ++eit) { 
        size_t src = boost::source(*eit, g_base);
        size_t tgt = boost::target(*eit, g_base);
        if (wmap[src] != 0 && wmap[tgt] == 0) {
          if (tgt != verts[vtxToVecIdx[sink]] && boost::out_degree(tgt, g_base) == 1 && !inf_path_to_sink[tgt] && (ssize_t) topo_sort_pos[tgt] < earliest) {
            earliest = topo_sort_pos[tgt];
            chosen = tgt;
          }
        }
      }
      if (earliest >= (ssize_t) topo_sort_pos.size()) {
        return MIN_CUT_FAILURE_SRC_INF_CAP_PATH;
      }

      // change all edges in the source partition to INF capacity, and add an edge from src->chosen with INF.
      for (auto& pr : myAdjList) {
        vertex_t src = pr.first;
        for (auto& pr2 : pr.second) {
          vertex_t dst = pr2.first;
          long long wt = pr2.second;

          if (wmap[vtxToVecIdx[src]] != 0 && wmap[vtxToVecIdx[dst]] != 0 && wt < INF_WT) {
            pr2.second = INF_WT;
          }
        }
      }
      myAdjList[source][boostToMyVtx[chosen]] = INF_WT;
    } else if (frac_src_cut > sourceWeight * (1+tol)) {
      //llvm::errs() << "collapse sink\n";

      ssize_t latest = -1;
      vertex_descriptor chosen;

      for (auto eit = orig_edges.first; eit != orig_edges.second; ++eit) { 
        size_t src = boost::source(*eit, g_base);
        size_t tgt = boost::target(*eit, g_base);
        if (wmap[src] != 0 && wmap[tgt] == 0) {
          if (src != verts[vtxToVecIdx[source]] && boost::out_degree(src, g_base) == 1 && !inf_path_to_source[src] && (ssize_t) topo_sort_pos[src] > latest) {
            latest = topo_sort_pos[boost::source(*eit, g_base)];
            chosen = boost::source(*eit, g_base);
          }
        }
      }
      if (latest < 0) {
        return MIN_CUT_FAILURE_SINK_INF_CAP_PATH;
      }

      // change all edges in the sink partition to INF capacity, and add an edge from chosen->sink with INF.
      for (auto& pr : myAdjList) {
        vertex_t src = pr.first;
        for (auto& pr2 : pr.second) {
          vertex_t dst = pr2.first;
          long long wt = pr2.second;

          if (wmap[vtxToVecIdx[src]] == 0 && wmap[vtxToVecIdx[dst]] == 0 && wt < INF_WT) {
            pr2.second = INF_WT;
          }
        }
      }
      myAdjList[boostToMyVtx[chosen]][sink] = INF_WT;
    } else {
      // figure out which nodes sit in which partition.
      for (size_t i = 0; i<verts.size(); ++i) {
        if (wmap[i] > 0) {
          // in source partition
          sourceSet.insert(boostToMyVtx[verts[i]]);
        }
      }
      llvm::errs() << "DONE\n";
      break;
    }
  }
  return MIN_CUT_SUCCESS;
}

bool pipelineHandler(ep2::FuncOp funcOp, PipelinePolicy* policy, PipelineResult* results) {
  if (funcOp.isExtern() || funcOp->getAttr("type").cast<StringAttr>().getValue() != "handler") {
    return false;
  }

  auto opToVertex = [](mlir::Operation* op) -> vertex_t { return (void*) op; };
  auto valueToVertex = [](mlir::Value v) -> vertex_t { return v.getAsOpaquePointer(); };

  std::unordered_map<vertex_t, std::unordered_map<vertex_t, long long>> myAdjList;
  std::unordered_map<vertex_t, mlir::Value> vertexToValue;
  std::unordered_map<vertex_t, mlir::Operation*> vertexToOp;

  std::unordered_map<mlir::Block*, vertex_t> sourceMap;
  std::unordered_map<mlir::Block*, vertex_t> sinkMap;
  size_t sourceSinkCtr = 0;
  vertex_t globalSource = (void*) sourceSinkCtr++;
  vertex_t globalSink = (void*) sourceSinkCtr++; 

  auto isSource = [&](vertex_t v) { return ((uintptr_t) v) < sourceSinkCtr && ((uintptr_t) v) % 2 == 0; };
  auto isSink = [&](vertex_t v) { return ((uintptr_t) v) < sourceSinkCtr && ((uintptr_t) v) % 2 == 1; };

  myAdjList[globalSource] = {};
  myAdjList[globalSink] = {};

  assert(funcOp->getNumRegions() == 1);
  mlir::Region& region = funcOp->getRegion(0);

  std::vector<std::pair<std::pair<vertex_t, vertex_t>, long long>> falseEdges;
  std::vector<std::pair<vertex_t, vertex_t>> pipelineConstraints;

  for (mlir::Block& blk : region.getBlocks()) {
    sourceMap[&blk] = (void*) sourceSinkCtr++;
    sinkMap[&blk] = (void*) sourceSinkCtr++;
    myAdjList[sourceMap[&blk]] = {};
    myAdjList[sinkMap[&blk]] = {};

    if (blk.hasNoPredecessors()) {
      falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{sourceMap[&blk], globalSource}, INF_WT);
    }
    if (blk.hasNoSuccessors()) {
      falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{globalSink, sinkMap[&blk]}, INF_WT);
    }
  }

  for (size_t i = 0; i<funcOp.getNumArguments(); ++i) {
    mlir::Value v = funcOp.getArgument(i);
    mlir::Type ty = v.getType();
    long long weight = policy->typeTransmitCost(ty);

    vertexToValue[valueToVertex(v)] = v;
    myAdjList[globalSource][valueToVertex(v)] = (long long) weight;
  }

  for (mlir::Block& blk : region.getBlocks()) {
    for (mlir::Block* pred : blk.getPredecessors()) {
      mlir::Operation* term = pred->getTerminator();
      assert(isa<cf::CondBranchOp>(term) || isa<cf::BranchOp>(term));
      for (size_t i = 0; i<blk.getNumArguments(); ++i) {
        mlir::Value v = blk.getArgument(i);
        vertexToOp[opToVertex(term)] = term;
        vertexToValue[valueToVertex(v)] = v;
        myAdjList[opToVertex(term)][valueToVertex(v)] = INF_WT;
      }
      falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{sourceMap[&blk], sinkMap[pred]}, INF_WT);
    }

    for (mlir::Operation& opr : blk.getOperations()) {
      mlir::Operation* op = &opr;
      // handle stateful operations
      if (isa<ep2::LookupOp>(op) || isa<ep2::UpdateOp>(op)) {
        mlir::Value table = isa<ep2::LookupOp>(op) ? cast<ep2::LookupOp>(op).getTable() : cast<ep2::UpdateOp>(op).getTable();
        mlir::Operation* tableDef = table.getDefiningOp();
        assert(tableDef != nullptr && "Table def must be in same block. TODO fix this");

        falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{op, tableDef}, INF_WT);
      }

      if (op->getNumResults() > 0) {
        for (size_t i = 0; i<op->getNumResults(); ++i) {
          mlir::Value v = op->getResult(i);
          if (!v.use_empty()) {
            mlir::Type ty = v.getType();
            long long weight = policy->typeTransmitCost(ty);
            if (isa<ep2::GlobalImportOp>(op)) {
              weight = INF_WT;
            }

            vertexToOp[opToVertex(op)] = op;
            vertexToValue[valueToVertex(v)] = v;
            assert(opToVertex(op) != valueToVertex(v));

            myAdjList[opToVertex(op)][valueToVertex(v)] = (long long) weight;
          }
        }
      } else {
        vertexToOp[opToVertex(op)] = op;
        falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{sinkMap[&blk], opToVertex(op)}, INF_WT);
        if (isa<ep2::ReturnOp>(op)) {
          pipelineConstraints.emplace_back(opToVertex(op), globalSink);
          myAdjList[opToVertex(op)][globalSink] = INF_WT;
        } else if (myAdjList.count(opToVertex(op)) == 0) {
          myAdjList[opToVertex(op)] = {};
        }
      }

      if (op->getNumOperands() > 0) {
        for (size_t i = 0; i<op->getNumOperands(); ++i) {
          Value v = op->getOperand(i);
          vertexToOp[opToVertex(op)] = op;
          vertexToValue[valueToVertex(v)] = v;
          myAdjList[valueToVertex(v)][opToVertex(op)] = INF_WT;

          if (v.getDefiningOp() != nullptr) {
            falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{opToVertex(op), opToVertex(v.getDefiningOp())}, INF_WT);
            pipelineConstraints.emplace_back(opToVertex(v.getDefiningOp()), opToVertex(op));
          } else {
            falseEdges.emplace_back(std::pair<vertex_t, vertex_t>{opToVertex(op), sourceMap[&blk]}, INF_WT);
            pipelineConstraints.emplace_back(sourceMap[&blk], opToVertex(op));
          }
        }
      } else {
        myAdjList[globalSource][opToVertex(op)] = SMALL_WT;
      }
    }
  }

  std::unordered_map<vertex_t, int> vtxWeights;
  for (const auto& pr : myAdjList) {
    if (isSource(pr.first) || isSink(pr.first)) {
      vtxWeights[pr.first] = 1;
    } else if (vertexToValue.find(pr.first) != vertexToValue.end()) {
      vtxWeights[pr.first] = policy->valueWeight(vertexToValue[pr.first]);
    } else if (vertexToOp.find(pr.first) != vertexToOp.end()) {
      vtxWeights[pr.first] = policy->operationWeight(vertexToOp[pr.first]);
    } else {
      llvm::errs() << pr.first << '\n';
      assert(false);
    }
  }

  std::unordered_map<vertex_t, std::string> vtxNames;
  for (const auto& pr : myAdjList) {
    if (vertexToOp.find(pr.first) != vertexToOp.end()) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      vertexToOp[pr.first]->print(ss);
      vtxNames[pr.first] = "OP " + ss.str();
    } else if (vertexToValue.find(pr.first) != vertexToValue.end()) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      vertexToValue[pr.first].print(ss);
      vtxNames[pr.first] = "VAL " + ss.str();
    } else if (isSource(pr.first)) {
      vtxNames[pr.first] = "SRC " + std::to_string(((uintptr_t) pr.first) / 2);
    } else if (isSink(pr.first)) {
      vtxNames[pr.first] = "SINK " + std::to_string(((uintptr_t) pr.first) / 2);
    } else {
      assert(false);
    }
  }

  for (const auto& pr : myAdjList) {
    for (const auto& pr2 : pr.second) {
      if (myAdjList.count(pr2.first) == 0) {
        llvm::errs() << vtxNames[pr.first] << '\n';
        llvm::errs() << pr.first << " " << pr2.first << '\n';
        if (vertexToOp.find(pr2.first) != vertexToOp.end()) {
          vertexToOp[pr2.first]->dump();
        } else if (vertexToValue.find(pr2.first) != vertexToValue.end()) {
          vertexToValue[pr2.first].dump();
        }
        assert(false);
      }
    }
  }

  // trick is to run it several times.
  int rc = 0;
  std::unordered_set<vertex_t> sourceSet;
  for (size_t i = 0; i<N_RAND_ITERS; ++i) {
    sourceSet.clear();
    rc = runBalancedMinCut(myAdjList, globalSource, globalSink, policy->tolerance, policy->sourceWeight, results->sourceWeight,
     sourceSet, falseEdges, vtxWeights, pipelineConstraints, vtxNames);
    if (rc == MIN_CUT_SUCCESS) {
      break;
    }
  } 
  if (rc > 0) {
    results->err = errMsgs[rc];
    return false;
  }

  if (policy->dumpCuts) {
    std::unordered_map<vertex_t, std::string> ids;
    std::unordered_map<std::string, size_t> idUniq;
    std::ofstream fout("cut.dot");
    fout << "digraph G {\n";
    for (const auto& pr : myAdjList) {
      if (ids.find(pr.first) == ids.end()) {
        if (pr.first == globalSource) {
          ids.emplace(pr.first, "globalSource");
        } else if (pr.first == globalSink) {
          ids.emplace(pr.first, "globalSink");
        } else if (isSource(pr.first)) {
          ids.emplace(pr.first, "src" + std::to_string(idUniq["src"]++));
        } else if (isSink(pr.first)) {
          ids.emplace(pr.first, "sink" + std::to_string(idUniq["sink"]++));
        } else if (vertexToOp.count(pr.first)) {
          ids.emplace(pr.first, opToId(vertexToOp[pr.first]) + 
            std::to_string(idUniq[opToId(vertexToOp[pr.first])]++));
        } else if (vertexToValue.count(pr.first)) {
          ids.emplace(pr.first, "v" + std::to_string(idUniq["v"]++));
        } else {
          assert(false && "Unhandled case");
        }
      }
      assert(vertexToOp.count(pr.first) || vertexToValue.count(pr.first) ||
        isSource(pr.first) || isSink(pr.first));
      fout << ids[pr.first];
      fout << " [ color = " << (sourceSet.count(pr.first) ? "red" : "blue") << " ];\n";
    }

    for (const auto& pr : myAdjList) {
      vertex_t s = pr.first;
      for (const auto& pr2 : pr.second) {
        vertex_t e = pr2.first;
        long long wt = pr2.second;
        assert(ids.find(s) != ids.end());
        fout << ids[s];
        fout << " -> ";
        if (s == globalSink) {
          fout << "gSink";
        } else {
          if (ids.find(e) == ids.end()) {
            llvm::errs() << "BOOM " << vtxNames[e] << '\n';
          }
          fout << ids[e];
        }
        fout << " [ label = " << (wt >= INF_MIN ? "INF" : std::to_string(wt)) << " ];\n";
      }
    }
    fout << "}\n";
    fout.flush();
    fout.close();
  }

  funcOp->walk([&](mlir::Operation* op) {
    if (isa<ep2::FuncOp>(op)) {
      return;
    }
    if (!sourceSet.count(opToVertex(op))) {
      results->sinkOps.insert(op);
      for (size_t i = 0; i<op->getNumResults(); ++i) {
        results->sinkValues.insert(op->getResult(i));
      }
    } else {
      for (size_t i = 0; i<op->getNumResults(); ++i) {
        if (!sourceSet.count(valueToVertex(op->getResult(i)))) {
          results->sinkValues.insert(op->getResult(i));
        }
      }
    }
  });
  return true;
}

struct NetronomeKCutPolicy : public PipelinePolicy {
  int numCuts;
  NetronomeKCutPolicy(int k, double tolerance = 0.1) : PipelinePolicy(1.0f / k, tolerance), numCuts(k) {}


  int valueWeight(mlir::Value v) override {
    return 1;
  }
  int operationWeight(mlir::Operation* op) override {
    return llvm::TypeSwitch<Operation *, int>(op)
        // non weight ops
        // .Case([&](ep2::GlobalImportOp) { return 100; })
        // .Case<ep2::StructAccessOp, ep2::ConstantOp,
        //       ep2::BitCastOp>([&](Operation *) { return 1; })
        .Default([&](Operation *) { return 1; });
  }

  int typeTransmitCost(mlir::Type ty) override {
    return 1 + typeTransmitCostRec(ty);
  }
  int typeTransmitCostRec(mlir::Type ty) {
    if (isa<ep2::StructType>(ty)) {
      int sz = 0;
      for (const auto& eTy : cast<ep2::StructType>(ty).getElementTypes()) {
        sz += typeTransmitCostRec(eTy);
      }
      return sz;
    } else if (isa<ep2::BufferType>(ty)) {
      // check definition in EmitNetronomePass.cpp (1 pointer, 2 ints)
      return 128;
    } else if (isa<ep2::TableType>(ty)) {
      return 64;
    } else if (isa<ep2::ContextType>(ty)) {
      return 64;
    } else if (isa<ep2::ContextRefType>(ty)) {
      return 64;
    } else if (isa<mlir::IntegerType>(ty)) {
      return cast<mlir::IntegerType>(ty).getWidth();
    } else if (isa<ep2::AtomType>(ty)) {
      return 32;
    } else {
      assert(false && "Support other types");
    }
  }

  std::pair<std::shared_ptr<PipelinePolicy>, std::shared_ptr<PipelinePolicy>> splitPolicy(PipelineResult &result) override {
    int newCuts = numCuts - 1;
    // also adjust tolerance

    double localTol = tolerance;
    localTol = tolerance / (1 - 1.0 / (numCuts));

    auto source = std::make_shared<NetronomeKCutPolicy>(newCuts, localTol);
    auto sink = std::make_shared<NetronomeKCutPolicy>(newCuts, localTol);

    bool noCutSource = result.sourceWeight < 1.0f / numCuts + localTol;
    source->done = newCuts == 1 || noCutSource;
    sink->done = newCuts == 1 || !noCutSource;
    return std::make_pair(source, sink);
  }

  std::pair<std::string, std::string> splitName() override {
    auto base = std::to_string(numCuts) + "Cut_";
    return std::make_pair(base + "source", base + "sink");
  }
};

using PolicyP = std::shared_ptr<PipelinePolicy>;
using SearchPair = std::pair<ep2::FuncOp, PolicyP>;
using SearchDirection = llvm::DenseMap<ep2::FuncOp, PolicyP>;

llvm::SmallVector<SearchDirection, 4> stepSearch(SearchDirection& sd) {
  SmallVector<SmallVector<SearchPair>> functionResult;

  PipelineResult results;
  bool allDone = true;
  SmallVector<SearchPair> choices;
  for (auto &[func, policy] : sd) {
    if (policy->done) {
      // llvm::errs() << "Done with " << func.getSymName() << '\n';
      // we are done with this funciton. continue;
      choices.push_back(std::make_pair(func, policy));
      continue;
    } else {
      allDone = false;
      // auto funcName = func.getSymName();
      // llvm::errs() << "Not done with " << func.getSymName() << '\n';
      if (pipelineHandler(func, policy.get(), &results)) {
        // llvm::errs() << "Min-cut suceeded for " << funcName << '\n';
        auto [sourceOp, sinkOp] = functionSplitter(func, results.sinkOps, results.sinkValues);
        auto [sourcePolicy, sinkPolicy] = policy->splitPolicy(results);
        // TODO: rename functions?
        auto [sourceName, sinkName] = policy->splitName();
        sourceOp.setSymName(func.getSymName().str() + "_" + sourceName);
        sinkOp.setSymName(func.getSymName().str() + "_" + sinkName);

        choices.push_back({sourceOp, sourcePolicy});
        choices.push_back({sinkOp, sinkPolicy});
      } else {
        llvm::errs() << "Min-cut failed for " << func.getSymName() << '\n';
        policy->done = true;
        choices.push_back(std::make_pair(func, policy));
      }
    }
  }
  if (!allDone)
    functionResult.push_back(std::move(choices));

  // construct return from generated functions.
  llvm::SmallVector<SearchDirection, 4> ret{};
  for (auto &funcChoices : functionResult) {
    auto &sd = ret.emplace_back();
    for (auto &[func, policy] : funcChoices) {
      sd[func] = policy;
    }
  }

  // llvm::errs() << "Returning " << ret.size() << '\n';

  return ret;
}

void kcutPolicy(Operation * moduleOp, int k) {
  SearchDirection sd;

  moduleOp->walk([&](ep2::FuncOp funcOp) {
    if (funcOp.isExtern() || !funcOp.isHandler())
      return;

    sd[funcOp] = std::make_shared<NetronomeKCutPolicy>(k, 0.1);
  });

  while (true) {
    auto cuts = stepSearch(sd);
    if (cuts.empty())
      break;
    sd = cuts[0];
  }

  llvm::errs() << "Finish kcut\n";
  for (auto &[func, policy] : sd) {
    llvm::errs() << "Function: " << func.getSymName() << '\n';
    llvm::errs() << "Policy: " << policy->sourceWeight << '\n';
  }

  moduleOp->walk([&](ep2::FuncOp funcOp){
    if (funcOp.isExtern() || !funcOp.isHandler())
      return;
    if (!sd.contains(funcOp))
      funcOp.erase();
  });
}

void bfsSearchPolicy(Operation * moduleOp) {
  std::queue<SearchDirection> cuts;

  // TODO: make a function for init cut
  // init cut: we try to cut the program into 1-3 pieces
  moduleOp->walk([&](ep2::FuncOp funcOp) {
    if (funcOp.isExtern() || !funcOp.isHandler())
      return;

    {
      llvm::DenseMap<ep2::FuncOp, PolicyP> sd;
      sd[funcOp] = std::make_shared<NetronomeKCutPolicy>(2);
      cuts.push(std::move(sd));
    }
    {
      llvm::DenseMap<ep2::FuncOp, PolicyP> sd;
      sd[funcOp] = std::make_shared<NetronomeKCutPolicy>(3);
      cuts.push(std::move(sd));
    }
  });

  // loop over cuts
  while (!cuts.empty()) {
    llvm::errs() << "Cuts size: " << cuts.size() << '\n';
    moduleOp->dump();
    llvm::errs() << "\n\n";


    auto next = cuts.front();
    cuts.pop();
    auto nexts = stepSearch(next);
    for (auto& n : nexts) {
      // exit if we think its good enough...
      cuts.push(n);
    }
  }

}

// Assumes all dead code is eliminated.
void PipelineHandlerPass::runOnOperation() {
  if (mode.getValue() == "search")
    bfsSearchPolicy(getOperation());
  else if (mode.getValue() == "kcut") {
    if (kNum.getValue() == 0) {
      llvm::errs() << "kcut mode requires kNum to be set\n";
      signalPassFailure();
      return;
    }

    kcutPolicy(getOperation(), kNum.getValue());
  } else {
    llvm::errs() << "Unknown mode: " << mode.getValue() << '\n';
    signalPassFailure();
  }
}

} // namespace ep2
} // namespace mlir
